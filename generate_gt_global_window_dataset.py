#!/usr/bin/env python3
"""
Build supervised labels for the *global* stack attention window in SpaceByte.

Unlike ``generate_gt_dataset.py`` (which searches a minimal **local** attention
window and toggles ``use_global_blocks``), this script **always keeps**
``use_global_blocks=True`` and searches the smallest **global** self-attention
window (``TransformerBlock.attention.attention_window`` on ``global_blocks``)
such that the greedy next-byte prediction at the target position still matches
either ``global_gt`` or ``real_gt``.

Definitions (per sample position, same data iterator as ``generate_gt_dataset``):

- ``real_gt``: target byte from the dataset at that position.
- ``global_gt``: greedy argmax at ``pos`` with global blocks ON, local window at
  the trained default, and **full** global self-attention (``attention_window=None``).
- ``pred_local_default``: greedy argmax with global blocks OFF (for analysis only).

Window search (``--window-search``):

- **linear** — scan downward from ``default_global_attention_window`` to ``1``.
- **binary** — assume monotonicity ``ok(w) ⇒ ok(w+1)`` and binary-search the
  smallest ``w`` in ``[1, default_global_attention_window]``.
- **compare** — run both; print diagnostics (like ``generate_gt_dataset``).
- **none** — skip search; store the default window and fraction ``1.0``.

``default_global_attention_window`` is ``max(1, min(global_context_size, context_size) - 1)``,
the largest meaningful ``attention_window`` index span on the compressed global
trajectory of length at most ``min(TG, Tx)``.

Outputs SQLite (table ``spacebyte_gt_global_window``) and/or JSON Lines.

Run from the ``spacebyte/`` repo root (same as ``validate.py``), with datasets prepared.

Examples::

  python generate_gt_global_window_dataset.py --checkpoint path/to/ckpt_best_loss.pt \\
    --output-db gt_global_win.sqlite --num-samples 1000 --split train --seed 0
"""

from __future__ import annotations

import argparse
import json
import os
import sqlite3
import sys
from pathlib import Path
from typing import Literal

import torch
from tqdm import tqdm

import util

_ROOT = Path(__file__).resolve().parent


def _sync_cuda_if_needed(device: str) -> None:
    if torch.cuda.is_available() and str(device).startswith("cuda"):
        torch.cuda.synchronize()


def set_spacebyte_local_attention_window(model: torch.nn.Module, window: int) -> None:
    w = int(window)
    assert w > 0
    model.config.local_attention_window = w
    for block in model.initial_blocks:
        block.attention.attention_window = w
    for block in model.final_blocks:
        block.attention.attention_window = w


def set_spacebyte_global_attention_window(model: torch.nn.Module, window: int | None) -> None:
    """Set causal attention span on **global** blocks only (compressed global stream)."""
    if window is not None:
        w = int(window)
        if w < 1:
            raise ValueError(f"global attention_window must be >= 1 or None, got {window!r}")
        for block in model.global_blocks:
            block.attention.attention_window = w
    else:
        for block in model.global_blocks:
            block.attention.attention_window = None


def greedy_next_token(
    model: torch.nn.Module,
    tokens: torch.Tensor,
    pos: int,
    *,
    use_global_blocks: bool,
    local_window: int,
    global_attn_window: int | None,
    autocast_ctx,
) -> int:
    """Single forward; argmax at ``pos`` (next-byte LM)."""
    prev_ug = model.config.use_global_blocks
    prev_loc_cfg = int(model.config.local_attention_window)
    prev_init_w = [b.attention.attention_window for b in model.initial_blocks]
    prev_fin_w = [b.attention.attention_window for b in model.final_blocks]
    prev_glb_w = [b.attention.attention_window for b in model.global_blocks]
    try:
        set_spacebyte_local_attention_window(model, local_window)
        set_spacebyte_global_attention_window(model, global_attn_window)
        model.config.use_global_blocks = bool(use_global_blocks)
        with torch.inference_mode():
            with autocast_ctx:
                logits, _ = model(tokens, targets=None)
        return int(logits[0, pos].argmax().item())
    finally:
        model.config.use_global_blocks = prev_ug
        model.config.local_attention_window = prev_loc_cfg
        for b, w in zip(model.initial_blocks, prev_init_w, strict=True):
            b.attention.attention_window = w
        for b, w in zip(model.final_blocks, prev_fin_w, strict=True):
            b.attention.attention_window = w
        for b, w in zip(model.global_blocks, prev_glb_w, strict=True):
            b.attention.attention_window = w


def matches_reference(pred: int, global_gt: int, real_gt: int) -> bool:
    return pred == global_gt or pred == real_gt


def default_global_attn_window(mc) -> int:
    """Largest index span on the global trajectory (length ≤ min(TG, byte context))."""
    tg = int(mc.global_context_size)
    tx = int(mc.context_size)
    g_len = min(tg, tx)
    return max(1, g_len - 1)


def _sanity_or_degenerate(
    model: torch.nn.Module,
    tokens: torch.Tensor,
    pos: int,
    default_local_window: int,
    default_global_window: int,
    global_gt: int,
    real_gt: int,
    margin: float,
    autocast_ctx,
) -> tuple[int, float] | tuple[()]:
    """If full global attention cannot match refs, return degenerate (default window, 1.0)."""
    pred_ref = greedy_next_token(
        model,
        tokens,
        pos,
        use_global_blocks=True,
        local_window=default_local_window,
        global_attn_window=None,
        autocast_ctx=autocast_ctx,
    )
    if not matches_reference(pred_ref, global_gt, real_gt):
        frac = min((float(default_global_window) + float(margin)) / float(default_global_window), 1.0)
        return default_global_window, frac
    return ()


def _min_global_window_linear(
    model: torch.nn.Module,
    tokens: torch.Tensor,
    pos: int,
    *,
    default_local_window: int,
    default_global_window: int,
    global_gt: int,
    real_gt: int,
    autocast_ctx,
) -> int:
    w_fail: int | None = None
    for w in range(default_global_window, 0, -1):
        pred = greedy_next_token(
            model,
            tokens,
            pos,
            use_global_blocks=True,
            local_window=default_local_window,
            global_attn_window=w,
            autocast_ctx=autocast_ctx,
        )
        if not matches_reference(pred, global_gt, real_gt):
            w_fail = w
            break
    if w_fail is None:
        return 1
    return w_fail + 1


def _min_global_window_binary(
    model: torch.nn.Module,
    tokens: torch.Tensor,
    pos: int,
    *,
    default_local_window: int,
    default_global_window: int,
    global_gt: int,
    real_gt: int,
    autocast_ctx,
) -> int:
    lo, hi = 1, default_global_window
    ans = default_global_window
    while lo <= hi:
        mid = (lo + hi) // 2
        pred = greedy_next_token(
            model,
            tokens,
            pos,
            use_global_blocks=True,
            local_window=default_local_window,
            global_attn_window=mid,
            autocast_ctx=autocast_ctx,
        )
        if matches_reference(pred, global_gt, real_gt):
            ans = mid
            hi = mid - 1
        else:
            lo = mid + 1
    return ans


def compute_min_global_window(
    model: torch.nn.Module,
    tokens: torch.Tensor,
    pos: int,
    *,
    default_local_window: int,
    default_global_window: int,
    global_gt: int,
    real_gt: int,
    margin: float,
    autocast_ctx,
    window_search: Literal["linear", "binary", "none"] = "linear",
) -> tuple[int, float, str]:
    if window_search == "none":
        return default_global_window, 1.0, window_search

    out = _sanity_or_degenerate(
        model,
        tokens,
        pos,
        default_local_window,
        default_global_window,
        global_gt,
        real_gt,
        margin,
        autocast_ctx,
    )
    if out:
        mw, frac = out
        return mw, frac, window_search

    if window_search == "linear":
        min_w = _min_global_window_linear(
            model,
            tokens,
            pos,
            default_local_window=default_local_window,
            default_global_window=default_global_window,
            global_gt=global_gt,
            real_gt=real_gt,
            autocast_ctx=autocast_ctx,
        )
    else:
        min_w = _min_global_window_binary(
            model,
            tokens,
            pos,
            default_local_window=default_local_window,
            default_global_window=default_global_window,
            global_gt=global_gt,
            real_gt=real_gt,
            autocast_ctx=autocast_ctx,
        )
    frac = min((float(min_w) + float(margin)) / float(default_global_window), 1.0)
    return min_w, frac, window_search


def compute_linear_binary_pair(
    model: torch.nn.Module,
    tokens: torch.Tensor,
    pos: int,
    *,
    default_local_window: int,
    default_global_window: int,
    global_gt: int,
    real_gt: int,
    margin: float,
    autocast_ctx,
) -> tuple[int, float, int, float]:
    out = _sanity_or_degenerate(
        model,
        tokens,
        pos,
        default_local_window,
        default_global_window,
        global_gt,
        real_gt,
        margin,
        autocast_ctx,
    )
    if out:
        mw, f = out
        return mw, f, mw, f

    min_lin = _min_global_window_linear(
        model,
        tokens,
        pos,
        default_local_window=default_local_window,
        default_global_window=default_global_window,
        global_gt=global_gt,
        real_gt=real_gt,
        autocast_ctx=autocast_ctx,
    )
    min_bin = _min_global_window_binary(
        model,
        tokens,
        pos,
        default_local_window=default_local_window,
        default_global_window=default_global_window,
        global_gt=global_gt,
        real_gt=real_gt,
        autocast_ctx=autocast_ctx,
    )
    frac_lin = min((float(min_lin) + float(margin)) / float(default_global_window), 1.0)
    frac_bin = min((float(min_bin) + float(margin)) / float(default_global_window), 1.0)
    return min_lin, frac_lin, min_bin, frac_bin


def _sqlite_table_columns(conn: sqlite3.Connection, table: str) -> set[str]:
    cur = conn.execute(f"PRAGMA table_info({table})")
    return {row[1] for row in cur.fetchall()}


def open_sqlite(path: str) -> sqlite3.Connection:
    conn = sqlite3.connect(path)
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS spacebyte_gt_global_window (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            split TEXT NOT NULL,
            sample_idx INTEGER NOT NULL,
            position INTEGER NOT NULL,
            real_gt INTEGER NOT NULL,
            global_gt INTEGER NOT NULL,
            pred_local_default INTEGER NOT NULL,
            use_global_blocks INTEGER NOT NULL,
            min_global_attention_window INTEGER NOT NULL,
            global_window_fraction REAL NOT NULL,
            checkpoint TEXT NOT NULL,
            default_global_attention_window INTEGER NOT NULL,
            default_local_attention_window INTEGER NOT NULL,
            window_search TEXT NOT NULL DEFAULT 'linear'
        )
        """
    )
    conn.commit()
    return conn


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build SpaceByte GT for minimal global-stack attention window."
    )
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--checkpoint-file", type=str, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--split", type=str, default="train")
    parser.add_argument("--num-samples", type=int, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--position",
        type=str,
        choices=("last", "all"),
        default="last",
        help="Which positions to label: last byte in context, or every valid position.",
    )
    parser.add_argument(
        "--margin",
        type=float,
        default=0.0,
        help="Added to min global window before dividing by default_global_attention_window.",
    )
    parser.add_argument("--output-db", type=str, default=None, help="SQLite database path.")
    parser.add_argument("--output-jsonl", type=str, default=None, help="Optional JSON Lines path.")
    parser.add_argument(
        "--window-search",
        type=str,
        choices=("linear", "binary", "compare", "none"),
        default="linear",
        help="Minimal global attention window: none, linear, binary (monotone ok(w)), or compare.",
    )
    args = parser.parse_args()

    if (
        args.window_search != "compare"
        and args.output_db is None
        and args.output_jsonl is None
    ):
        print("error: specify at least one of --output-db or --output-jsonl", file=sys.stderr)
        sys.exit(1)

    os.chdir(_ROOT)

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    import megabyte  # noqa: F401
    import transformer  # noqa: F401
    from megabyte import MegaByte  # noqa: F401
    from spacebyte import SpaceByte, SpaceByteConfig  # noqa: F401
    from transformer import Transformer  # noqa: F401

    from validate import build_model, prepare_model_config, resolve_checkpoint_path

    import data as data_mod

    ckpt_path = resolve_checkpoint_path(args.checkpoint, args.checkpoint_file)
    if not os.path.isfile(ckpt_path):
        print(f"error: checkpoint file not found: {ckpt_path}", file=sys.stderr)
        sys.exit(1)

    checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
    Model = eval(checkpoint["model"])
    raw_cfg = dict(checkpoint["model_config"])
    for k in list(raw_cfg):
        if not hasattr(Model.Config, k):
            del raw_cfg[k]

    cfg = prepare_model_config(Model, raw_cfg, {})
    Model_cls = Model
    cfg_obj = Model_cls.Config(**cfg)
    if not isinstance(cfg_obj, SpaceByteConfig):
        print("error: this script only supports SpaceByte checkpoints.", file=sys.stderr)
        sys.exit(1)

    model, mc = build_model(Model, cfg, checkpoint["state_dict"], device)
    default_local_window = int(mc.local_attention_window)
    default_global_window = default_global_attn_window(mc)

    tc = checkpoint["train_config"]
    d = data_mod.dataset(tc["dataset"], mc.tokenizer)
    model.dataset_tokenizer = d.tokenizer

    context_size = mc.context_size
    autocast = util.autocast_context(tc["dtype"])

    if args.split not in d.splits():
        print(f"error: split {args.split!r} not in dataset splits {list(d.splits())}", file=sys.stderr)
        sys.exit(1)

    data_iter = d.iter(
        args.split,
        context_size=context_size,
        batch_size=1,
        seed=args.seed,
        device=device,
    )

    conn: sqlite3.Connection | None = None
    jsonl_f = None
    rows_sql: list[tuple] = []

    cmp_n = 0
    cmp_sum_abs_win = 0.0
    cmp_sum_win_norm = 0.0
    cmp_sum_gt_frac = 0.0
    cmp_sum_win_lin = 0.0
    cmp_sum_win_bin = 0.0

    if args.output_db:
        conn = open_sqlite(args.output_db)
    if args.output_jsonl:
        jsonl_f = open(args.output_jsonl, "w", encoding="utf-8")

    sample_idx = 0
    n_target = args.num_samples

    try:
        pbar = tqdm(total=n_target, desc="GT global-window samples", leave=True)
        while sample_idx < n_target:
            tokens, targets = next(data_iter)
            B, T = targets.shape
            assert B == 1

            if args.position == "last":
                positions = [T - 1]
            else:
                positions = list(range(T))

            for pos in positions:
                if sample_idx >= n_target:
                    break
                rt = int(targets[0, pos].item())
                if rt < 0:
                    continue

                global_gt = greedy_next_token(
                    model,
                    tokens,
                    pos,
                    use_global_blocks=True,
                    local_window=default_local_window,
                    global_attn_window=None,
                    autocast_ctx=autocast,
                )

                pred_local_default = greedy_next_token(
                    model,
                    tokens,
                    pos,
                    use_global_blocks=False,
                    local_window=default_local_window,
                    global_attn_window=None,
                    autocast_ctx=autocast,
                )

                if args.window_search == "compare":
                    min_lin, frac_lin, min_bin, frac_bin = compute_linear_binary_pair(
                        model,
                        tokens,
                        pos,
                        default_local_window=default_local_window,
                        default_global_window=default_global_window,
                        global_gt=global_gt,
                        real_gt=rt,
                        margin=args.margin,
                        autocast_ctx=autocast,
                    )
                    min_w = min_lin
                    frac = frac_lin
                    ws = "compare"

                    d_win_abs = abs(min_lin - min_bin)
                    d_win_norm = d_win_abs / float(default_global_window) if default_global_window > 0 else 0.0
                    d_gt_frac = abs(frac_lin - frac_bin)

                    cmp_n += 1
                    cmp_sum_abs_win += d_win_abs
                    cmp_sum_win_norm += d_win_norm
                    cmp_sum_gt_frac += d_gt_frac
                    cmp_sum_win_lin += float(min_lin)
                    cmp_sum_win_bin += float(min_bin)

                    mean_abs = cmp_sum_abs_win / cmp_n
                    mean_wn = cmp_sum_win_norm / cmp_n
                    mean_gt = cmp_sum_gt_frac / cmp_n
                    mean_win_lin = cmp_sum_win_lin / cmp_n
                    mean_win_bin = cmp_sum_win_bin / cmp_n

                    print(
                        f"compare[{sample_idx}] "
                        f"win_linear={int(min_lin)} win_binary={int(min_bin)} "
                        f"|d_win|={d_win_abs} d_win_norm={d_win_norm:.6f} "
                        f"d_gt_frac={d_gt_frac:.6f} | "
                        f"mean_win_linear={mean_win_lin:.2f} mean_win_binary={mean_win_bin:.2f} "
                        f"mean_|d_win|={mean_abs:.4f} mean_d_win_norm={mean_wn:.6f} "
                        f"mean_d_gt_frac={mean_gt:.6f} (n={cmp_n})",
                        flush=True,
                    )
                else:
                    min_w, frac, ws = compute_min_global_window(
                        model,
                        tokens,
                        pos,
                        default_local_window=default_local_window,
                        default_global_window=default_global_window,
                        global_gt=global_gt,
                        real_gt=rt,
                        margin=args.margin,
                        autocast_ctx=autocast,
                        window_search=args.window_search,
                    )

                row = {
                    "split": args.split,
                    "sample_idx": sample_idx,
                    "position": pos,
                    "real_gt": rt,
                    "global_gt": global_gt,
                    "pred_local_default": pred_local_default,
                    "use_global_blocks": 1,
                    "min_global_attention_window": min_w,
                    "global_window_fraction": frac,
                    "checkpoint": ckpt_path,
                    "default_global_attention_window": default_global_window,
                    "default_local_attention_window": default_local_window,
                    "window_search": ws,
                }

                if args.window_search == "compare" and jsonl_f:
                    row["min_global_attention_window_linear"] = min_lin
                    row["min_global_attention_window_binary"] = min_bin
                    row["global_window_fraction_linear"] = frac_lin
                    row["global_window_fraction_binary"] = frac_bin
                    row["compare_abs_d_win"] = d_win_abs
                    row["compare_d_win_norm"] = d_win_norm
                    row["compare_d_gt_frac"] = d_gt_frac

                if jsonl_f:
                    jsonl_f.write(json.dumps(row, ensure_ascii=False) + "\n")

                if conn:
                    rows_sql.append(
                        (
                            row["split"],
                            row["sample_idx"],
                            row["position"],
                            row["real_gt"],
                            row["global_gt"],
                            row["pred_local_default"],
                            row["use_global_blocks"],
                            row["min_global_attention_window"],
                            row["global_window_fraction"],
                            row["checkpoint"],
                            row["default_global_attention_window"],
                            row["default_local_attention_window"],
                            row["window_search"],
                        )
                    )

                sample_idx += 1
                pbar.update(1)

        pbar.close()

        if args.window_search == "compare" and cmp_n > 0:
            print(
                f"compare summary (n={cmp_n}): "
                f"mean_win_linear={cmp_sum_win_lin / cmp_n:.2f} "
                f"mean_win_binary={cmp_sum_win_bin / cmp_n:.2f} "
                f"mean_|d_win|={cmp_sum_abs_win / cmp_n:.4f} "
                f"mean_d_win_norm={cmp_sum_win_norm / cmp_n:.6f} "
                f"mean_d_gt_frac={cmp_sum_gt_frac / cmp_n:.6f}",
                flush=True,
            )

        if conn and rows_sql:
            conn.executemany(
                """
                INSERT INTO spacebyte_gt_global_window (
                    split, sample_idx, position, real_gt, global_gt, pred_local_default,
                    use_global_blocks, min_global_attention_window, global_window_fraction,
                    checkpoint, default_global_attention_window, default_local_attention_window,
                    window_search
                ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?)
                """,
                rows_sql,
            )
            conn.commit()

    finally:
        if jsonl_f:
            jsonl_f.close()
        if conn:
            conn.close()

    _sync_cuda_if_needed(device)
    if args.output_db or args.output_jsonl:
        print(f"Wrote {sample_idx} rows.", flush=True)
    else:
        print(f"Processed {sample_idx} samples.", flush=True)


if __name__ == "__main__":
    main()
