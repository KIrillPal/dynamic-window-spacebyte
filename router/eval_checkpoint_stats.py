#!/usr/bin/env python3
"""Load a router checkpoint and report classification / regression distributions on val."""

from __future__ import annotations

import argparse
import dataclasses
import math
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

_SPACEBYTE_ROOT = Path(__file__).resolve().parents[1]
if str(_SPACEBYTE_ROOT) not in sys.path:
    sys.path.insert(0, str(_SPACEBYTE_ROOT))

from router.config import RouterTrainConfig  # noqa: E402
from router.dataset import CombinedSpaceByteRouterDataset, SourceSpec  # noqa: E402
from router.model import RouterModelProtocol, build_router_model  # noqa: E402
from router.train import resolve_db_specs, resolve_tokenizer_name  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Validation stats for AdaptiveComputeRouter checkpoint.")
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--limit-samples", type=int, default=None, help="Cap val dataset size for quick runs.")
    p.add_argument(
        "--regression-figure",
        type=str,
        default=None,
        help="If set, save PNG with overlaid histograms of GT vs predicted regression (matplotlib).",
    )
    return p.parse_args()


def config_from_saved_dict(raw: dict) -> RouterTrainConfig:
    names = {f.name for f in dataclasses.fields(RouterTrainConfig)}
    data = {k: v for k, v in raw.items() if k in names}
    cfg = RouterTrainConfig(**data)
    cfg.normalize_learning_rates()
    return cfg


def resolve_relative_db_paths(cfg: RouterTrainConfig, root: Path) -> None:
    """Make sqlite paths absolute against ``root`` (spacebyte/) when they are relative."""

    def abs_sqlite(p: str) -> str:
        path = Path(p)
        if path.is_absolute():
            return str(path)
        return str((root / path).resolve())

    if cfg.db_paths:
        out: list = []
        for item in cfg.db_paths:
            if isinstance(item, str):
                out.append(abs_sqlite(item))
            elif isinstance(item, dict):
                d = dict(item)
                key = "path" if "path" in d else "db_path"
                if key in d:
                    d[key] = abs_sqlite(str(d[key]))
                out.append(d)
            else:
                out.append(item)
        cfg.db_paths = out
    elif cfg.db_path:
        cfg.db_path = abs_sqlite(cfg.db_path)
    if cfg.classification_only_db_paths:
        cfg.classification_only_db_paths = [abs_sqlite(p) for p in cfg.classification_only_db_paths]


def maybe_align_classifier_head(model: RouterModelProtocol, state_dict: dict[str, torch.Tensor]) -> None:
    """Replace last Linear if checkpoint has 1 output logit (legacy) vs current 2-class head."""
    if "classifier.4.weight" not in state_dict:
        return
    w = state_dict["classifier.4.weight"]
    out_dim, hidden = int(w.shape[0]), int(w.shape[1])
    current = model.classifier[-1]
    if not isinstance(current, nn.Linear):
        raise TypeError("Expected classifier[-1] to be Linear")
    if current.out_features == out_dim:
        return
    model.classifier[-1] = nn.Linear(hidden, out_dim)


def move_batch(batch: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    return {k: v.to(device, non_blocking=True) for k, v in batch.items()}


def classification_pred_prob(logits: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """pred label float {0,1}, prob of positive class."""
    if logits.dim() == 2 and logits.size(-1) == 2:
        prob_pos = torch.softmax(logits, dim=-1)[:, 1]
        pred = logits.argmax(dim=-1).float()
        return pred, prob_pos
    z = logits.squeeze(-1)
    prob_pos = torch.sigmoid(z)
    pred = (prob_pos >= 0.5).float()
    return pred, prob_pos


def save_regression_histogram_figure(path: Path, pred: np.ndarray, gt: np.ndarray, *, title: str) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    bins = np.linspace(0, 1, 41)
    ax0, ax1 = axes
    ax0.hist(gt, bins=bins, color="tab:blue", alpha=0.85, edgecolor="white", linewidth=0.5)
    ax0.set_title("Ground truth (local_window_fraction)")
    ax0.set_xlim(0, 1)
    ax0.set_xlabel("fraction")
    ax0.set_ylabel("count")

    ax1.hist(pred, bins=bins, color="tab:orange", alpha=0.85, edgecolor="white", linewidth=0.5)
    ax1.set_title("Predicted (window_ratio)")
    ax1.set_xlim(0, 1)
    ax1.set_xlabel("fraction")
    ax1.set_ylabel("count")

    fig.suptitle(title)
    fig.tight_layout()

    fig2, ax = plt.subplots(figsize=(7, 4))
    ax.hist(
        gt,
        bins=bins,
        alpha=0.55,
        label="GT",
        color="tab:blue",
        density=True,
        edgecolor="white",
        linewidth=0.5,
    )
    ax.hist(
        pred,
        bins=bins,
        alpha=0.55,
        label="Prediction",
        color="tab:orange",
        density=True,
        edgecolor="white",
        linewidth=0.5,
    )
    ax.set_xlim(0, 1)
    ax.set_xlabel("fraction [0, 1]")
    ax.set_ylabel("density")
    ax.legend()
    ax.set_title(title + " (normalized overlap)")
    fig2.tight_layout()

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    stem = path.stem
    out_side = path.with_name(f"{stem}_side_by_side.png")
    out_overlap = path.with_name(f"{stem}_density_overlap.png")
    fig.savefig(out_side, dpi=150)
    fig2.savefig(out_overlap, dpi=150)
    plt.close(fig)
    plt.close(fig2)
    print(f"Saved regression histograms: {out_side}, {out_overlap}")


def print_histogram(name: str, values: np.ndarray, bins: int = 10, range_: tuple[float, float] = (0.0, 1.0)) -> None:
    values = values[np.isfinite(values)]
    if values.size == 0:
        print(f"  {name}: (no samples)")
        return
    hist, edges = np.histogram(values, bins=bins, range=range_)
    total = max(int(hist.sum()), 1)
    print(f"  {name} (n={values.size}), bins {bins} on {range_}:")
    for i in range(len(hist)):
        lo, hi = edges[i], edges[i + 1]
        bar = "#" * max(1, int(50 * hist[i] / total))
        print(f"    [{lo:.3f}, {hi:.3f}): {hist[i]:6d} ({100.0 * hist[i] / total:5.1f}%) {bar}")


@torch.inference_mode()
def main() -> None:
    args = parse_args()
    os.chdir(_SPACEBYTE_ROOT)

    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    print("=== checkpoint ===")
    print(" path:", args.checkpoint)
    print(" global_step:", ckpt.get("global_step"))
    print(" best_metric:", ckpt.get("best_metric"))
    raw_cfg = ckpt["config"]
    print(
        " saved keys (subset): model_type=",
        raw_cfg.get("model_type", "minilm"),
        "model_name=",
        raw_cfg.get("model_name"),
        "tokenizer_name=",
        raw_cfg.get("tokenizer_name"),
        "freeze_encoder=",
        raw_cfg.get("freeze_encoder"),
    )

    cfg = config_from_saved_dict(raw_cfg)
    resolve_relative_db_paths(cfg, _SPACEBYTE_ROOT)

    tokenizer = AutoTokenizer.from_pretrained(resolve_tokenizer_name(cfg))
    tokenizer.truncation_side = cfg.tokenizer_truncation_side

    source = None
    if cfg.source_dataset or cfg.source_tokenizer or cfg.source_context_size:
        if cfg.source_dataset is None or cfg.source_context_size is None:
            raise ValueError("Checkpoint expects explicit source_dataset and source_context_size.")
        source = SourceSpec(
            dataset=cfg.source_dataset,
            tokenizer=cfg.source_tokenizer,
            context_size=cfg.source_context_size,
        )

    val_seed = cfg.val_generation_seed if cfg.val_generation_seed is not None else cfg.generation_seed
    val_specs = resolve_db_specs(cfg, default_seed=val_seed)
    val_dataset = CombinedSpaceByteRouterDataset(
        db_specs=val_specs,
        split=cfg.val_split,
        hf_tokenizer=tokenizer,
        source=source,
        max_length=cfg.max_length,
        limit=args.limit_samples,
    )

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    loader = DataLoader(
        val_dataset,
        batch_size=cfg.eval_batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=device.type == "cuda",
    )

    model = build_router_model(cfg, tokenizer)
    sd = ckpt["model_state_dict"]
    maybe_align_classifier_head(model, sd)
    missing, unexpected = model.load_state_dict(sd, strict=False)
    if missing:
        print("WARNING load_state_dict missing:", missing[:8], "..." if len(missing) > 8 else "")
    if unexpected:
        print("WARNING load_state_dict unexpected:", unexpected[:8], "..." if len(unexpected) > 8 else "")

    model.to(device)
    model.eval()
    if cfg.freeze_encoder:
        model.freeze_encoder(unfreeze_last_n_layers=cfg.unfreeze_last_n_layers)

    cls_true_list: list[np.ndarray] = []
    cls_pred_list: list[np.ndarray] = []
    prob_pos_list: list[np.ndarray] = []
    reg_mask_list: list[np.ndarray] = []
    reg_pred_list: list[np.ndarray] = []
    reg_true_list: list[np.ndarray] = []

    for batch in loader:
        batch = move_batch(batch, device)
        outputs = model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            token_type_ids=batch.get("token_type_ids"),
        )
        logits = outputs["global_logits"]
        pred, prob_pos = classification_pred_prob(logits)
        y = batch["use_global_blocks"].float().reshape(-1).cpu().numpy()
        cls_true_list.append(y)
        cls_pred_list.append(pred.cpu().numpy().reshape(-1))
        prob_pos_list.append(prob_pos.cpu().numpy().reshape(-1))

        ratio = outputs["window_ratio"].cpu().numpy().reshape(-1)
        rt = batch["local_window_fraction"].cpu().numpy().reshape(-1)
        mask = batch["has_regression"].cpu().numpy().reshape(-1) > 0.5
        reg_mask_list.append(mask)
        reg_pred_list.append(ratio)
        reg_true_list.append(rt)

    y_true = np.concatenate(cls_true_list)
    y_pred = np.concatenate(cls_pred_list)
    prob_pos = np.concatenate(prob_pos_list)

    print()
    print("=== classification (val) ===")
    print(" samples:", y_true.size)
    print(" true label=1 rate:", float(np.mean(y_true)))
    print(" pred label=1 rate:", float(np.mean(y_pred)))
    print(" accuracy:", float(np.mean(y_true == y_pred)))
    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    tn = int(np.sum((y_true == 0) & (y_pred == 0)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))
    print(" confusion [ [TN FP]\n           [FN TP] ]:")
    print(f"           [ [{tn:6d} {fp:6d}]")
    print(f"             [{fn:6d} {tp:6d}] ]")
    print_histogram("P(class=1)", prob_pos, bins=10, range_=(0.0, 1.0))

    mask_all = np.concatenate(reg_mask_list)
    pred_all = np.concatenate(reg_pred_list)
    true_all = np.concatenate(reg_true_list)
    reg_idx = mask_all
    print()
    print("=== regression (val, has_regression only) ===")
    print(" samples with regression label:", int(reg_idx.sum()), "/", mask_all.size)
    if reg_idx.any():
        pr = pred_all[reg_idx]
        tr = true_all[reg_idx]
        err = pr - tr
        print(" pred mean / std / min / max:", float(pr.mean()), float(pr.std()), float(pr.min()), float(pr.max()))
        print(" target mean / std / min / max:", float(tr.mean()), float(tr.std()), float(tr.min()), float(tr.max()))
        print(" MAE:", float(np.abs(err).mean()))
        print(" RMSE:", float(math.sqrt(np.mean(err**2))))
        print_histogram("predicted window_ratio", pr, bins=10, range_=(0.0, 1.0))
        print_histogram("target local_window_fraction", tr, bins=10, range_=(0.0, 1.0))
        print_histogram("error (pred - target)", err, bins=10, range_=(-1.0, 1.0))
        if args.regression_figure:
            title = f"val regression (n={pr.size}) — {Path(args.checkpoint).name}"
            save_regression_histogram_figure(Path(args.regression_figure), pr, tr, title=title)
    else:
        print(" (no rows with regression — classification-only DB or NULL fractions)")


if __name__ == "__main__":
    main()
