#!/usr/bin/env python3
"""
Generate supervised labels for SpaceByte routing / adaptive local attention:

1) use_global_blocks ∈ {0,1} — whether global transformer blocks should be used for
   this position (1 = locals alone cannot match ``global_gt`` / ``real_gt`` under greedy argmax).

2) local_window_fraction ∈ [0,1] — min((T + margin) / trained_local_attention_window, 1),
   where T is the smallest local attention window (integer ≥ 1) for which, under the
   chosen global-blocks mode, the greedy argmax at the target position still matches
   either ``global_gt`` or ``real_gt``.

Procedure (per sample position):
  - ``real_gt``: target byte from the dataset at that position.
  - Forward with global blocks ON and default ``local_attention_window`` → ``global_gt``.
  - Forward with global blocks OFF, same window → ``pred_local``.
  - If ``pred_local`` equals ``global_gt`` or ``real_gt``, then ``use_global_blocks = 0``.
    Otherwise ``use_global_blocks = 1``.
  - Window search uses the same ``use_global_blocks`` flag. Strategies (see
    ``--window-search``):

    * **linear** — scan downward from the default window (no extra assumptions).
    * **binary** — assume **monotonicity**: if greedy prediction matches the reference
      at window ``w``, it also matches at ``w+1`` (larger local window only adds
      context). Then the minimal matching window is the smallest ``w`` in
      ``[1, default_window]`` with a match, found in ``O(log default_window)`` forwards.
    * **compare** — run both linear and binary on each sample, print per-step and
      cumulative normalized differences (window size and GT fraction); optional outputs
      use linear min-window labels when files are given.
    * **none** — skip local-window search; compute only ``use_global_blocks`` and store
      the default ``local_attention_window`` / fraction ``1.0``.

Outputs an SQLite database (--output-db) and optionally JSON Lines (--output-jsonl), except
in **compare** mode without paths (terminal-only).

Run from the ``spacebyte/`` repo root (same as ``validate.py``), with datasets prepared.

Examples::

  python generate_gt_dataset.py --checkpoint path/to/run/ckpt_best_loss.pt \\
    --output-db gt.sqlite --num-samples 1000 --split train --seed 0

  python generate_gt_dataset.py --checkpoint ckpt.pt --num-samples 500 \\
    --window-search compare
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
    """Patch only local (byte-level) blocks; leave global stack windows unchanged."""
    w = int(window)
    assert w > 0
    model.config.local_attention_window = w
    for block in model.initial_blocks:
        block.attention.attention_window = w
    for block in model.final_blocks:
        block.attention.attention_window = w


def greedy_next_token(
    model: torch.nn.Module,
    tokens: torch.Tensor,
    pos: int,
    *,
    use_global_blocks: bool,
    local_window: int,
    autocast_ctx,
) -> int:
    """Single forward; argmax at ``pos`` (causal LM next-byte prediction)."""
    set_spacebyte_local_attention_window(model, local_window)
    prev = model.config.use_global_blocks
    model.config.use_global_blocks = bool(use_global_blocks)
    try:
        with torch.inference_mode():
            with autocast_ctx:
                logits, _ = model(tokens, targets=None)
        return int(logits[0, pos].argmax().item())
    finally:
        model.config.use_global_blocks = prev


def matches_reference(pred: int, global_gt: int, real_gt: int) -> bool:
    return pred == global_gt or pred == real_gt


def _sanity_or_degenerate(
    model: torch.nn.Module,
    tokens: torch.Tensor,
    pos: int,
    default_window: int,
    global_gt: int,
    real_gt: int,
    pred_local_default: int,
    margin: float,
    autocast_ctx,
) -> tuple[int, int, float] | tuple[int, bool]:
    """
    If reference forward under the chosen global-blocks mode fails at full window,
    return degenerate (use_global_blocks, default_window, fraction).

    Otherwise return (use_global_blocks, use_global_bool) for window-size search.
    """
    if matches_reference(pred_local_default, global_gt, real_gt):
        use_global_blocks = 0
    else:
        use_global_blocks = 1

    use_global = use_global_blocks == 1
    pred_ref = greedy_next_token(
        model,
        tokens,
        pos,
        use_global_blocks=use_global,
        local_window=default_window,
        autocast_ctx=autocast_ctx,
    )
    if not matches_reference(pred_ref, global_gt, real_gt):
        min_window = default_window
        frac = min((float(min_window) + float(margin)) / float(default_window), 1.0)
        return use_global_blocks, min_window, frac

    return use_global_blocks, use_global


def _min_local_window_linear(
    model: torch.nn.Module,
    tokens: torch.Tensor,
    pos: int,
    *,
    use_global_blocks: bool,
    default_window: int,
    global_gt: int,
    real_gt: int,
    autocast_ctx,
) -> int:
    """Scan ``default_window .. 1``; minimal window is one past first failing size."""
    w_fail: int | None = None
    for w in range(default_window, 0, -1):
        pred = greedy_next_token(
            model,
            tokens,
            pos,
            use_global_blocks=use_global_blocks,
            local_window=w,
            autocast_ctx=autocast_ctx,
        )
        if not matches_reference(pred, global_gt, real_gt):
            w_fail = w
            break

    if w_fail is None:
        return 1
    return w_fail + 1


def _min_local_window_binary(
    model: torch.nn.Module,
    tokens: torch.Tensor,
    pos: int,
    *,
    use_global_blocks: bool,
    default_window: int,
    global_gt: int,
    real_gt: int,
    autocast_ctx,
) -> int:
    """
    Find smallest ``w`` in ``[1, default_window]`` where prediction matches refs.

    Assumption (monotone correctness): let ``ok(w)`` mean the greedy prediction at
    ``pos`` matches ``global_gt`` or ``real_gt``. We assume ``ok(w)`` implies
    ``ok(w+1)`` (more local attention context cannot break a previously correct argmax).
    Under that, the set of good ``w`` is a suffix ``[w_min, default_window]``.
    """
    lo, hi = 1, default_window
    ans = default_window
    while lo <= hi:
        mid = (lo + hi) // 2
        pred = greedy_next_token(
            model,
            tokens,
            pos,
            use_global_blocks=use_global_blocks,
            local_window=mid,
            autocast_ctx=autocast_ctx,
        )
        if matches_reference(pred, global_gt, real_gt):
            ans = mid
            hi = mid - 1
        else:
            lo = mid + 1
    return ans


def compute_use_global_and_min_window(
    model: torch.nn.Module,
    tokens: torch.Tensor,
    pos: int,
    default_window: int,
    global_gt: int,
    real_gt: int,
    pred_local_default: int,
    *,
    margin: float,
    autocast_ctx,
    window_search: Literal["linear", "binary", "none"] = "linear",
) -> tuple[int, int, float, str]:
    """
    Returns:
      use_global_blocks: 1 if global blocks are necessary (local-only misses both refs),
                         0 if local-only already hits global_gt or real_gt.
      min_window: smallest integer window ≥ 1 where prediction still matches refs,
                  under the chosen ``use_global_blocks`` mode.
      local_window_fraction: min((min_window + margin) / default_window, 1.0).
      window_search: ``linear``, ``binary`` or ``none`` (echo of request; binary assumes match(w) ⇒ match(w+1)).
    """
    if window_search == "none":
        use_global_blocks = 0 if matches_reference(pred_local_default, global_gt, real_gt) else 1
        return use_global_blocks, default_window, 1.0, window_search

    out = _sanity_or_degenerate(
        model,
        tokens,
        pos,
        default_window,
        global_gt,
        real_gt,
        pred_local_default,
        margin,
        autocast_ctx,
    )
    if len(out) == 3:
        use_global_blocks, min_window, frac = out
        return use_global_blocks, min_window, frac, window_search

    use_global_blocks, use_global = out

    if window_search == "linear":
        min_window = _min_local_window_linear(
            model,
            tokens,
            pos,
            use_global_blocks=use_global,
            default_window=default_window,
            global_gt=global_gt,
            real_gt=real_gt,
            autocast_ctx=autocast_ctx,
        )
    else:
        min_window = _min_local_window_binary(
            model,
            tokens,
            pos,
            use_global_blocks=use_global,
            default_window=default_window,
            global_gt=global_gt,
            real_gt=real_gt,
            autocast_ctx=autocast_ctx,
        )

    frac = min((float(min_window) + float(margin)) / float(default_window), 1.0)
    return use_global_blocks, min_window, frac, window_search


def compute_linear_binary_pair(
    model: torch.nn.Module,
    tokens: torch.Tensor,
    pos: int,
    default_window: int,
    global_gt: int,
    real_gt: int,
    pred_local_default: int,
    *,
    margin: float,
    autocast_ctx,
) -> tuple[int, int, float, int, float]:
    """
    Shared sanity check, then minimal windows via linear scan and binary search.

    Returns:
      use_global_blocks, min_linear, frac_linear, min_binary, frac_binary.
    """
    out = _sanity_or_degenerate(
        model,
        tokens,
        pos,
        default_window,
        global_gt,
        real_gt,
        pred_local_default,
        margin,
        autocast_ctx,
    )
    if len(out) == 3:
        use_global_blocks, mw, f = out
        return use_global_blocks, mw, f, mw, f

    use_global_blocks, use_global = out
    min_lin = _min_local_window_linear(
        model,
        tokens,
        pos,
        use_global_blocks=use_global,
        default_window=default_window,
        global_gt=global_gt,
        real_gt=real_gt,
        autocast_ctx=autocast_ctx,
    )
    min_bin = _min_local_window_binary(
        model,
        tokens,
        pos,
        use_global_blocks=use_global,
        default_window=default_window,
        global_gt=global_gt,
        real_gt=real_gt,
        autocast_ctx=autocast_ctx,
    )
    frac_lin = min((float(min_lin) + float(margin)) / float(default_window), 1.0)
    frac_bin = min((float(min_bin) + float(margin)) / float(default_window), 1.0)
    return use_global_blocks, min_lin, frac_lin, min_bin, frac_bin


def _sqlite_table_columns(conn: sqlite3.Connection, table: str) -> set[str]:
    cur = conn.execute(f"PRAGMA table_info({table})")
    return {row[1] for row in cur.fetchall()}


def open_sqlite(path: str) -> sqlite3.Connection:
    conn = sqlite3.connect(path)
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS spacebyte_gt (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            split TEXT NOT NULL,
            sample_idx INTEGER NOT NULL,
            position INTEGER NOT NULL,
            real_gt INTEGER NOT NULL,
            global_gt INTEGER NOT NULL,
            pred_local_default INTEGER NOT NULL,
            use_global_blocks INTEGER NOT NULL,
            min_local_window INTEGER NOT NULL,
            local_window_fraction REAL NOT NULL,
            checkpoint TEXT NOT NULL,
            default_local_attention_window INTEGER NOT NULL,
            window_search TEXT NOT NULL DEFAULT 'linear'
        )
        """
    )
    cols = _sqlite_table_columns(conn, "spacebyte_gt")
    if "window_search" not in cols:
        conn.execute("ALTER TABLE spacebyte_gt ADD COLUMN window_search TEXT NOT NULL DEFAULT 'linear'")
    conn.commit()
    return conn

def main() -> None:
    parser = argparse.ArgumentParser(description="Build SpaceByte GT SQLite / JSONL.")
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
        help="Added to min window before dividing by trained local_attention_window.",
    )
    parser.add_argument("--output-db", type=str, default=None, help="SQLite database path.")
    parser.add_argument("--output-jsonl", type=str, default=None, help="Optional JSON Lines path.")
    parser.add_argument(
        "--window-search",
        type=str,
        choices=("linear", "binary", "compare", "none"),
        default="linear",
        help="Minimal local window: none (only use_global_blocks; keep default window), "
        "linear scan, binary search (monotone ok(w)), or compare (run both; print "
        "per-step and mean |Δwin|, Δwin_norm, Δgt_frac to stdout).",
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
    default_window = int(mc.local_attention_window)

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
        pbar = tqdm(total=n_target, desc="GT samples", leave=True)
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
                    local_window=default_window,
                    autocast_ctx=autocast,
                )

                pred_local_default = greedy_next_token(
                    model,
                    tokens,
                    pos,
                    use_global_blocks=False,
                    local_window=default_window,
                    autocast_ctx=autocast,
                )

                if args.window_search == "compare":
                    ug, min_lin, frac_lin, min_bin, frac_bin = compute_linear_binary_pair(
                        model,
                        tokens,
                        pos,
                        default_window,
                        global_gt,
                        rt,
                        pred_local_default,
                        margin=args.margin,
                        autocast_ctx=autocast,
                    )
                    min_w = min_lin
                    frac = frac_lin
                    ws = "compare"

                    d_win_abs = abs(min_lin - min_bin)
                    d_win_norm = (
                        d_win_abs / float(default_window) if default_window > 0 else 0.0
                    )
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

                    row = {
                        "split": args.split,
                        "sample_idx": sample_idx,
                        "position": pos,
                        "real_gt": rt,
                        "global_gt": global_gt,
                        "pred_local_default": pred_local_default,
                        "use_global_blocks": ug,
                        "min_local_window": min_w,
                        "local_window_fraction": frac,
                        "checkpoint": ckpt_path,
                        "default_local_attention_window": default_window,
                        "window_search": ws,
                    }
                    if jsonl_f:
                        row["min_local_window_linear"] = min_lin
                        row["min_local_window_binary"] = min_bin
                        row["local_window_fraction_linear"] = frac_lin
                        row["local_window_fraction_binary"] = frac_bin
                        row["compare_abs_d_win"] = d_win_abs
                        row["compare_d_win_norm"] = d_win_norm
                        row["compare_d_gt_frac"] = d_gt_frac
                else:
                    ug, min_w, frac, ws = compute_use_global_and_min_window(
                        model,
                        tokens,
                        pos,
                        default_window,
                        global_gt,
                        rt,
                        pred_local_default,
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
                        "use_global_blocks": ug,
                        "min_local_window": min_w,
                        "local_window_fraction": frac,
                        "checkpoint": ckpt_path,
                        "default_local_attention_window": default_window,
                        "window_search": ws,
                    }

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
                            row["min_local_window"],
                            row["local_window_fraction"],
                            row["checkpoint"],
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
                INSERT INTO spacebyte_gt (
                    split, sample_idx, position, real_gt, global_gt, pred_local_default,
                    use_global_blocks, min_local_window, local_window_fraction,
                    checkpoint, default_local_attention_window, window_search
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
