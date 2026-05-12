# Adaptive Compute Router

Small two-head router for SpaceByte GT labels:

- `use_global_blocks`: binary classification head.
- `local_window_fraction`: regression head in `[0, 1]`.

The default backbone is `sentence-transformers/all-MiniLM-L6-v2`. The model uses
`last_hidden_state` with masked mean pooling, then separate MLP heads. As a
lighter second option, set `model_type: bilstm` to use token embeddings plus a
bidirectional LSTM encoder with the same two heads. For BiLSTM, `model_name`
may be `null`, but `tokenizer_name` must still point to a HuggingFace tokenizer
because the dataset stores text prefixes as token IDs.

## Data

The current GT SQLite stores labels plus `sample_idx`/`position`, but not the
prefix text. `SpaceByteRouterDataset` replays the original deterministic
SpaceByte dataset iterator using the checkpoint path stored in SQLite. For the
existing `gt_trainval_v1.sqlite`, this reconstructs PG-19 prefixes generated
with `seed=0` and `position=last`.

Prefixes are decoded to text and then tokenized by MiniLM with
`max_length=384` and left truncation, keeping the most recent context.

## Single GPU

```bash
cd /home/kondrashov_k/mipt/hw/nlp/spacebyte
python3 -m router.train --config router/config.default.yaml
```

Multiple SQLite databases can be passed either in YAML:

```yaml
db_paths:
  - path: /path/to/gt_train.sqlite
    seed: 0
  - path: /path/to/gt_val.sqlite
    seed: 42
  - path: /path/to/gt_test.sqlite
    seed: 42
```

or via CLI:

```bash
python3 -m router.train \
  --config router/config.default.yaml \
  --db-paths gt_trainval_v1.sqlite gt_test_v1.sqlite
```

`CombinedSpaceByteRouterDataset` drops exact duplicate `(sample_idx, position)`
rows inside each split and generation seed, and raises on conflicting duplicate
labels. Rows with the same `sample_idx` but different seeds are treated as
different prefixes.

Classification-only SQLite files are supported. If `local_window_fraction` is
missing or `NULL`, the row gets `has_regression=0`: it contributes to the BCE
classification loss and classification metrics, while regression loss, MSE and
MAE ignore it.

If a DB still has numeric placeholders in `local_window_fraction` (e.g. window
size instead of a fraction), mark the database as classification-only so those
values are ignored:

```yaml
db_paths:
  - path: gt_only_bin_v2.sqlite
    seed: 42
    classification_only: true
```

Or list absolute paths once:

```yaml
classification_only_db_paths:
  - /home/kondrashov_k/mipt/hw/nlp/spacebyte/gt_only_bin_v2.sqlite
```

Useful overrides:

```bash
python3 -m router.train \
  --config router/config.default.yaml \
  --epochs 3 \
  --train-batch-size 32 \
  --learning-rate 3e-5 \
  --set scheduler=cosine \
  --set mlflow=true
```

BiLSTM run on the train/val SQLite:

```bash
python3 -m router.train --config router/config.bilstm.yaml
```

## Two GPUs

```bash
cd /home/kondrashov_k/mipt/hw/nlp/spacebyte
# If you see NCCL / localhost socket errors (IPv6), force IPv4:
export MASTER_ADDR=127.0.0.1
torchrun --nproc_per_node=2 -m router.train --config router/config.default.yaml
```

## Outputs

The trainer writes:

- `config.resolved.yaml`
- `checkpoint_best.pt`
- `checkpoint_last.pt`

When MLflow is enabled, it logs train/val loss, accuracy, precision, recall, F1,
MSE and MAE.

Validation metrics are also printed after every epoch.
