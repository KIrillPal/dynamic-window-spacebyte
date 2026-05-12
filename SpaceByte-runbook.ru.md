# SpaceByte: метод, обучение, валидация и метрики

Документ описывает архитектуру **SpaceByte** в этом репозитории, устройство **`train.py`** и **`validate.py`**, смысл основных метрик и типовые команды запуска. Исходная статья: [SpaceByte: Towards Deleting Tokenization from Large Language Modeling](https://arxiv.org/abs/2404.14408).

---

## 1. Что такое SpaceByte

**SpaceByte** — байтовая (без отдельного BPE/SentencePiece на входе) языковая модель с **двумя масштабами**: локальный трансформер работает в пространстве размерности `d_local` на полной длине контекста в **байтах** (`context_size`), а **глобальный** стек — в размерности `d_model` на сжатой последовательности длины до **`global_context_size`**.

Идея «пространственного» сжатия: не каждый байт попадает в глобальный уровень. В режиме **`patch_method=utf8`** (основной в репозитории) по правилам UTF‑8 и границам символов отмечаются позиции, где нужен **глобальный** контекст; из них строится до `TG = global_context_size` «глобальных шагов» на батч. Остальная информация смешивается через **локальные** блоки с окном **`local_attention_window`**.

Стек в `spacebyte.py` (упрощённо):

1. **Эмбеддинг** байтов + локальные позиционные параметры.
2. **`initial_blocks`** — локальные `TransformerBlock` (`n_initial_layers` слоёв).
3. **Подъём в глобальное пространство**: из локальных представлений по индексам `global_ts` собирается тензор длины `global_T[b] ≤ TG`, проецируется в `d_model`, добавляются глобальные позиции.
4. **`global_blocks`** — глобальные трансформер-слои (можно отключить флагом **`use_global_blocks=False`**: тогда глобальные блоки в forward не вызываются, но часть учёта FLOPs в коде может оставаться — см. предупреждение в `validate.py`).
5. **Запись обратно** в локальную траекторию (`index_add`), затем **`final_blocks`**.
6. **`logits`** — предсказание следующего байта (vocab байтов, `tokenizer is None` в конфиге).

В лоссах для UTF‑8 у части позиций target может быть **`ignore_index=-1`** (хвост за пределом числа глобальных слотов) — метрика **`ignored fraction`**.

---

## 2. Файл `train.py`: как устроен

### 2.1 Точка входа

```bash
python train.py --dataset=pg19 --model=SpaceByte ...
```

Используется **Fire**: все именованные аргументы попадают в **`TrainConfig`** (поля `TrainConfig` совпадают с именами флагов). Лишние ключи могут попадать в `TrainConfig.args` и далее в overrides конфига модели.

Функция **`train(**kwargs)`** (в конце файла):

- при необходимости инициализирует **distributed** (`setup_distributed`);
- в цикле с уменьшением **`micro_batch_size`** при **OOM** пересобирает тренер;
- создаёт **`Train(model_config, train_config)`** и вызывает **`trainer.train()`**.

### 2.2 `TrainConfig`

Содержит: **`model`**, **`dataset`**, **`batch_size`**, **`micro_batch_size`**, **`iters`** (строка с формулой, например `'30e9/tokens'`), **`lr`**, **`dtype`**, **`device`**, **`out_dir`**, интервалы **`eval_interval`**, **`eval_iters`**, **`final_eval_iters`**, wandb, чекпоинты и т.д. Числовые поля вроде `iters` / `eval_iters` могут вычисляться через **`eval()`** из строк (с подстановкой `B`, `tokens`, `N`, …).

### 2.3 Класс `Train`

- **`Train.from_args` / конструктор**: строит датасет `data.dataset`, модель `eval(c.model)(model_config)`, оптимизатор, autocast, опционально **`torch.compile`**, DDP.
- **`dataset_iter(split)`** — итератор батчей: `context_size`, `micro_batch_size`, `seed` зависят от сплита.
- **`train()`** — основной цикл по **`iter_num`**: для каждого шага делается **`n_micro_batches = batch_size / (micro_batch_size * world_size)`** микро-батчей, **`loss.backward()`**, **`grad_clip`**, шаг оптимизатора, логирование, периодически **`estimate_loss`** на val/train, чекпоинты в **`out_dir`** (копируются `.py` файлы проекта, stdout в `stdout.txt`).
- **`estimate_losses`** / внешняя функция **`estimate_loss`**: несколько шагов инференса, усреднение лоссов через **`util.MeanError`**, для байтовой модели добавляется **`bits per byte`** из **`cross entropy`** (натуральные единицы → биты на байт).

### 2.4 Метрики в `estimate_loss` (`train.py`)

На каждом микро-батче модель возвращает словарь **`losses`** (для SpaceByte с UTF‑8: как минимум **`cross entropy`**, **`loss`**, **`global context`**, **`ignored fraction`**, плюс **`token_XE`** добавляется в `estimate_loss`).

После усреднения по **`eval_iters`**:

| Имя | Смысл |
|-----|--------|
| **`cross entropy`** | Средняя кросс-энтропия по позициям (натуральные единицы, не нормированная на ln2). |
| **`cross entropy` ± `… stat`** | Оценка среднего и разброса (дисперсия по микро-батчам; мало **`eval_iters`** даёт нестабильные `stat`). |
| **`bits per byte`** | `cross_entropy / (bytes_per_token * ln 2)` — биты на байт (для байтового LM **`bytes_per_token=1`**). |
| **`global context`** | Среднее по батчу **`global_T`**: сколько глобальных позиций реально использовано (не константа **`global_context_size`**, а фактическое число после правил патчинга). |
| **`ignored fraction`** | Доля позиций в target, равных **`-1`** (игнорируются в CE). |
| **`token_XE`** | Дополнительная сводка по батчу (batch-level CE). |

**Важно:** в `util.MeanError` для дисперсии используется деление на **`n-1`**; при **`eval_iters < 3`** возможны проблемы — в **`validate.py`** это принудительно поднимается до **3**.

---

## 3. Файл `validate.py`: как устроен

Скрипт **не обучает** модель: загружает чекпоинт, считает лоссы и производительность, опционально генерирует примеры текста и пишет **YAML**.

### 3.1 Порядок работы `main()`

1. Разбор **argparse**, при необходимости **`eval_iters = max(3, …)`**.
2. **`chdir`** в корень пакета (рядом с `datasets/`).
3. Загрузка **`torch.load(checkpoint)`**: строка **`model`** → **`eval` → класс**, **`model_config`**, **`state_dict`**, **`train_config`**.
4. **Overrides** конфига: **`--local-attention-window`**, **`--use-global-blocks` / `--no-global-blocks`** (для экспериментов без глобального mixing).
5. **`build_model`**: `Model.Config(**cfg)`, веса, **`model.eval()`**.
6. Для **SpaceByte**: теоретические **`flops_parts`** (`_spacebyte_theoretical_flops_parts`), **`code_flops_per_token`** из **`model.n_flops()`**.
7. Датасет как при обучении: **`data.dataset(train_config["dataset"], mc.tokenizer)`**.
8. Таблицы: конфиг прогона, параметры vs данные, блок **theoretical FLOPs**, блок **code accounting**.
9. Цикл по **`--splits`**: для каждого сплита **`estimate_loss`** (тот же код, что в `train.py`), сбор строки в **сводную таблицу** (метрики по строкам, сплиты по столбцам после **транспонирования**).
10. Опционально **`--examples`**: промпты из случайных строк корпуса по сплитам, продолжение через **`model.generate`**, прогресс **`tqdm`** (или **`print`**, если **`--no-tqdm`**).
11. Печать полного текстового отчёта; при **`--output-yaml`** — **`yaml.safe_dump`** отчёта (числа округлены).

### 3.2 Сводная таблица и FLOPs (кратко)

- **`max FLOPs/B`** — номинальная теоретическая оценка FLOPs **на один сырой байт** (полный глобальный путь по формуле в `_spacebyte_theoretical_flops_parts`).
- **`FLOPs/B`** — то же в **пересчёте на утилизацию глобального контекста**: локальная часть + **`Global%`** × глобальная часть, на байт (**`Global%`** ≈ **100 × mean(global_T) / global_context_size**).
- **`max FLOPs/s`**, **`FLOPs/s`** — полные FLOPs за оценочный прогон **делённые на wall time** (эквивалентно «FLOPs/B × logits/s» при **`bytes_per_token=1`**).
- **`real FLOPs/B`** — **`n_flops / context_size`**, приведённое к байту через **`bytes_per_token`**.

Подробные формулы ведущих членов — в функциях **`_table5_leading_m_global_local`**, **`_spacebyte_theoretical_flops_parts`**, **`_adjusted_flops_per_position`**.

### 3.3 Основные флаги `validate.py`

| Флаг | Назначение |
|------|------------|
| **`--checkpoint`** | Путь к файлу **`.pt`** или к **каталогу** прогона (тогда ищется **`--checkpoint-file`**, по умолчанию **`ckpt.pt`**). |
| **`--device`** | Например **`cuda:0`** (по умолчанию CUDA при наличии). |
| **`--eval-iters`** | Число микро-батчей на сплит (минимум **3**). |
| **`--splits`** | Список через запятую, например **`train,val,test`** (имена как в **`data.dataset`**). |
| **`--local-attention-window`**, **`--no-global-blocks`** | Переопределения конфига модели на инференсе. |
| **`--no-tqdm`** | Без progress bar в **`estimate_loss`** и без tqdm при генерации примеров. |
| **`--examples`**, **`--examples-seed`**, **`--examples-gen-tokens`**, … | Число и параметры **генерации** продолжений от модели. |
| **`--output-yaml`** | Полный отчёт в YAML. |

---

## 4. Как запускать

Рабочая директория — корень репозитория **`spacebyte/`** (там же лежат `train.py`, `validate.py`, каталог **`datasets/`** при локальной подготовке данных).

### 4.1 Зависимости и данные

```bash
pip install -r requirements.txt
```

Подготовка датасетов (пример для pg19 в UTF‑8) — см. **`reproduce/README.md`**, например:

```bash
python prepare.py pg19 --out_dir=datasets/pg19
```

### 4.2 Обучение (пример)

Полный сет гиперпараметров — в **`reproduce/README.md`**. Минимальный пример вызова:

```bash
python train.py --dataset=pg19 --model=SpaceByte --context_size=2048 \
  --global_context_size=1024 --d_model=1024 --d_local=384 --n_layers=12 --n_local_layers=12 \
  --local_attention_window=384 --batch_size=64 --micro_batch_size=4 --iters=1000 \
  --out_dir=my-run
```

Чекпоинты и логи появятся в подкаталоге вида **`my-run/Train--.../`**.

### 4.3 Валидация чекпоинта

```bash
python validate.py --checkpoint path/to/run/ckpt_best_loss.pt --eval-iters 50 \
  --splits train,val,test --output-yaml report.yaml
```

Каталог с несколькими файлами:

```bash
python validate.py --checkpoint path/to/run --checkpoint-file ckpt_best_loss.pt
```

Отключить генерацию примеров: **`--examples 0`**.

---

## 5. Связь с README и воспроизведением

- Краткое описание репозитория: **`README.md`**.
- Скрипты и команды масштабных прогонов из статьи: **`reproduce/README.md`**, **`reproduce/jobs.py`**, **`reproduce/plots.py`**.

Этот runbook дополняет их с точки зрения **кода** `train.py` / `validate.py` и **интерпретации метрик**.
