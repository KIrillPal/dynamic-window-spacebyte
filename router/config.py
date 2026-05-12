from __future__ import annotations

import dataclasses
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass
class RouterTrainConfig:
    # Data
    db_path: str = "/home/kondrashov_k/mipt/hw/nlp/spacebyte/gt_trainval_v1.sqlite"
    # Accepts either ["a.sqlite", ...] or [{"path": "a.sqlite", "seed": 42}, ...].
    db_paths: list[Any] | None = None
    #: Paths that should train classification only (ignore local_window_fraction in SQLite).
    classification_only_db_paths: list[str] | None = None
    train_split: str = "train"
    val_split: str = "val"
    source_dataset: str | None = None
    source_tokenizer: str | None = None
    source_context_size: int | None = None
    generation_seed: int = 0
    train_generation_seed: int | None = None
    val_generation_seed: int | None = None
    max_length: int = 384
    tokenizer_truncation_side: str = "left"
    num_workers: int = 0
    max_train_samples: int | None = None
    max_val_samples: int | None = None

    # Model
    model_type: str = "minilm"
    model_name: str | None = "sentence-transformers/all-MiniLM-L6-v2"
    tokenizer_name: str | None = None
    dropout: float = 0.1
    hidden_dropout: float = 0.1
    regression_activation: str = "sigmoid"
    lstm_embedding_dim: int = 128
    lstm_hidden_size: int = 256
    lstm_num_layers: int = 2
    freeze_encoder: bool = False
    unfreeze_last_n_layers: int | None = None
    gradient_checkpointing: bool = False
    #: Train only classifier (freeze regressor; loss = classification only). CLI: --cls-only
    train_classifier_only: bool = False
    #: Train only regressor (freeze classifier; loss = regression only). CLI: --reg-only
    train_regressor_only: bool = False

    # Optimization
    epochs: int = 5
    train_batch_size: int = 16
    eval_batch_size: int = 32
    gradient_accumulation_steps: int = 1
    learning_rate: float = 2e-5
    encoder_learning_rate: float | None = None
    head_learning_rate: float | None = None
    weight_decay: float = 0.01
    adam_beta1: float = 0.9
    adam_beta2: float = 0.999
    adam_eps: float = 1e-8
    max_grad_norm: float = 1.0
    scheduler: str = "linear"
    warmup_steps: int = 0
    warmup_ratio: float = 0.06
    num_cycles: float = 0.5

    # Losses / metrics
    classification_loss_weight: float = 1.0
    regression_loss_weight: float = 1.0
    regression_loss: str = "mse"
    huber_delta: float = 0.05
    #: Extra weight for positive class (label 1) in softmax cross-entropy; None = uniform classes.
    pos_weight: float | None = None

    # Runtime
    seed: int = 0
    device: str | None = None
    precision: str = "bf16"
    log_every_steps: int = 20
    eval_every_steps: int = 0
    save_every_steps: int = 0
    output_dir: str = "router_runs/adaptive_compute_router"
    save_best_metric: str = "val/loss"
    save_best_mode: str = "min"
    resume_from: str | None = None

    # MLflow
    mlflow: bool = False
    mlflow_tracking_uri: str | None = None
    mlflow_experiment: str = "spacebyte-router"
    mlflow_run_name: str | None = None

    def __post_init__(self) -> None:
        self.normalize_learning_rates()

    def normalize_learning_rates(self) -> None:
        """Coerce LR fields after YAML/CLI overrides (strings break LambdaLR)."""
        self.learning_rate = _coerce_float(self.learning_rate, "learning_rate")
        if self.encoder_learning_rate is not None:
            self.encoder_learning_rate = _coerce_float(self.encoder_learning_rate, "encoder_learning_rate")
        if self.head_learning_rate is not None:
            self.head_learning_rate = _coerce_float(self.head_learning_rate, "head_learning_rate")

    def validate_head_training_flags(self) -> None:
        if self.train_classifier_only and self.train_regressor_only:
            raise ValueError(
                "train_classifier_only and train_regressor_only cannot both be true "
                "(do not combine --cls-only and --reg-only)."
            )

    @classmethod
    def from_file(cls, path: str | Path | None) -> "RouterTrainConfig":
        if path is None:
            return cls()

        path = Path(path)
        data: dict[str, Any]
        with path.open("r", encoding="utf-8") as f:
            if path.suffix.lower() in {".yaml", ".yml"}:
                data = yaml.safe_load(f) or {}
            elif path.suffix.lower() == ".json":
                data = json.load(f)
            else:
                raise ValueError(f"Unsupported config format: {path}")

        field_names = {f.name for f in dataclasses.fields(cls)}
        data = {k: v for k, v in data.items() if k in field_names}
        return cls(**data)

    def update_from_pairs(self, pairs: list[str]) -> None:
        for pair in pairs:
            if "=" not in pair:
                raise ValueError(f"Expected KEY=VALUE override, got {pair!r}")
            key, value = pair.split("=", 1)
            if not hasattr(self, key):
                raise ValueError(f"Unknown config key: {key}")
            setattr(self, key, _parse_value(value))

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as f:
            yaml.safe_dump(self.to_dict(), f, sort_keys=True, allow_unicode=False)


def _coerce_float(value: Any, field_name: str) -> float:
    if isinstance(value, bool) or value is None:
        raise TypeError(f"{field_name} must be a number, got {value!r}")
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        return float(value.strip())
    raise TypeError(f"{field_name} must be float-compatible, got {type(value).__name__}: {value!r}")


def _parse_value(value: str) -> Any:
    lower = value.lower()
    if lower in {"none", "null"}:
        return None
    if lower == "true":
        return True
    if lower == "false":
        return False
    if value[:1] in {"[", "{"}:
        return yaml.safe_load(value)
    try:
        return int(value)
    except ValueError:
        pass
    try:
        return float(value)
    except ValueError:
        return value
