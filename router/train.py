#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from transformers import AutoTokenizer

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from router.config import RouterTrainConfig  # noqa: E402
from router.dataset import CombinedSpaceByteRouterDataset, DbSpec, SourceSpec  # noqa: E402
from router.model import build_router_model  # noqa: E402
from router.trainer import RouterTrainer, seed_everything  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Adaptive Compute Router.")
    parser.add_argument("--config", type=str, default=None, help="YAML/JSON config path.")
    parser.add_argument("--db-path", type=str, default=None, help="GT SQLite path.")
    parser.add_argument(
        "--db-paths",
        nargs="+",
        default=None,
        help="One or more GT SQLite paths. Overrides --db-path/config db_path.",
    )
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--model-type", choices=("minilm", "bilstm"), default=None)
    parser.add_argument("--model-name", type=str, default=None)
    parser.add_argument("--tokenizer-name", type=str, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--train-batch-size", type=int, default=None)
    parser.add_argument("--eval-batch-size", type=int, default=None)
    parser.add_argument("--learning-rate", type=float, default=None)
    parser.add_argument("--scheduler", type=str, default=None)
    parser.add_argument("--precision", choices=("fp32", "fp16", "bf16"), default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--mlflow", action="store_true", help="Enable MLflow logging.")
    parser.add_argument(
        "--set",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="Override any RouterTrainConfig field, can be repeated.",
    )
    head_mode = parser.add_mutually_exclusive_group()
    head_mode.add_argument(
        "--cls-only",
        action="store_true",
        help="Train only the classification head (freeze regressor; optimization uses cls loss only).",
    )
    head_mode.add_argument(
        "--reg-only",
        action="store_true",
        help="Train only the regression head (freeze classifier; optimization uses regression loss only).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = RouterTrainConfig.from_file(args.config)
    apply_named_overrides(cfg, args)
    cfg.update_from_pairs(args.set)
    cfg.normalize_learning_rates()
    if args.cls_only:
        cfg.train_classifier_only = True
    if args.reg_only:
        cfg.train_regressor_only = True
    cfg.validate_head_training_flags()
    seed_everything(cfg.seed)

    tokenizer_name = resolve_tokenizer_name(cfg)
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    tokenizer.truncation_side = cfg.tokenizer_truncation_side

    source = None
    if cfg.source_dataset or cfg.source_tokenizer or cfg.source_context_size:
        if cfg.source_dataset is None or cfg.source_context_size is None:
            raise ValueError(
                "When overriding source data, set at least source_dataset and source_context_size."
            )
        source = SourceSpec(
            dataset=cfg.source_dataset,
            tokenizer=cfg.source_tokenizer,
            context_size=cfg.source_context_size,
        )

    train_db_specs = resolve_db_specs(
        cfg,
        default_seed=cfg.train_generation_seed
        if cfg.train_generation_seed is not None
        else cfg.generation_seed,
    )
    val_db_specs = resolve_db_specs(
        cfg,
        default_seed=cfg.val_generation_seed
        if cfg.val_generation_seed is not None
        else cfg.generation_seed,
    )
    train_dataset = CombinedSpaceByteRouterDataset(
        db_specs=train_db_specs,
        split=cfg.train_split,
        hf_tokenizer=tokenizer,
        source=source,
        max_length=cfg.max_length,
        limit=cfg.max_train_samples,
    )
    val_dataset = CombinedSpaceByteRouterDataset(
        db_specs=val_db_specs,
        split=cfg.val_split,
        hf_tokenizer=tokenizer,
        source=source,
        max_length=cfg.max_length,
        limit=cfg.max_val_samples,
    )

    model = build_router_model(cfg, tokenizer)
    trainer = RouterTrainer(cfg, model, train_dataset, val_dataset)
    trainer.train()


def apply_named_overrides(cfg: RouterTrainConfig, args: argparse.Namespace) -> None:
    mapping = {
        "db_path": args.db_path,
        "output_dir": args.output_dir,
        "model_type": args.model_type,
        "model_name": args.model_name,
        "tokenizer_name": args.tokenizer_name,
        "epochs": args.epochs,
        "train_batch_size": args.train_batch_size,
        "eval_batch_size": args.eval_batch_size,
        "learning_rate": args.learning_rate,
        "scheduler": args.scheduler,
        "precision": args.precision,
        "device": args.device,
    }
    for key, value in mapping.items():
        if value is not None:
            setattr(cfg, key, value)
    if args.mlflow:
        cfg.mlflow = True
    if args.db_paths is not None:
        cfg.db_paths = args.db_paths


def resolve_tokenizer_name(cfg: RouterTrainConfig) -> str:
    name = cfg.tokenizer_name or cfg.model_name
    if name is None:
        raise ValueError(
            "tokenizer_name must be set when model_name is null. "
            "For BiLSTM, keep model_name: null and set tokenizer_name."
        )
    return name


def resolve_db_specs(cfg: RouterTrainConfig, *, default_seed: int) -> list[DbSpec]:
    cls_only_set = _classification_only_path_set(cfg)

    def flag_for_path(path_str: str, explicit: bool | None) -> bool:
        if explicit is not None:
            return explicit
        resolved = Path(path_str).expanduser().resolve()
        return resolved in cls_only_set

    if cfg.db_paths:
        specs = []
        for item in cfg.db_paths:
            if isinstance(item, str):
                path_str = str(item)
                specs.append(
                    DbSpec(
                        path=path_str,
                        generation_seed=default_seed,
                        classification_only=flag_for_path(path_str, None),
                    )
                )
            elif isinstance(item, dict):
                path = item.get("path") or item.get("db_path")
                if path is None:
                    raise ValueError(f"db_paths item misses 'path': {item!r}")
                path_str = str(path)
                seed = item.get("seed", item.get("generation_seed", default_seed))
                explicit = item.get("classification_only")
                if explicit is None:
                    explicit = item.get("cls_only")
                if explicit is None:
                    explicit = item.get("binary_only")
                if explicit is not None:
                    explicit = bool(explicit)
                specs.append(
                    DbSpec(
                        path=path_str,
                        generation_seed=int(seed),
                        classification_only=flag_for_path(path_str, explicit),
                    )
                )
            else:
                raise TypeError(f"Unsupported db_paths item: {item!r}")
        return specs
    if "," in cfg.db_path:
        return [
            DbSpec(
                path=path.strip(),
                generation_seed=default_seed,
                classification_only=flag_for_path(path.strip(), None),
            )
            for path in cfg.db_path.split(",")
            if path.strip()
        ]
    return [
        DbSpec(
            path=cfg.db_path,
            generation_seed=default_seed,
            classification_only=flag_for_path(cfg.db_path, None),
        )
    ]


def _classification_only_path_set(cfg: RouterTrainConfig) -> set[Path]:
    raw = cfg.classification_only_db_paths or []
    return {Path(p).expanduser().resolve() for p in raw}


if __name__ == "__main__":
    main()
