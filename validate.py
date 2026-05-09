#!/usr/bin/env python3
"""
Evaluate a SpaceByte checkpoint on pg19-style dataset splits.

Examples:
  python validate.py --checkpoint path/to/run/ckpt_best_loss.pt
  python validate.py --checkpoint ckpt.pt --output-yaml results.yaml
  python validate.py --checkpoint ckpt.pt --splits train,val,test --examples 5 --examples-gen-tokens 200
"""

from __future__ import annotations

import argparse
import math
import os
import random
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from tqdm import tqdm

import util

_ROOT = Path(__file__).resolve().parent

DECIMALS = 3

_SPLIT_FILES = {"train": "train.txt", "val": "validation.txt", "test": "test.txt"}


def _sync_cuda_if_needed(device: str) -> None:
    if torch.cuda.is_available() and str(device).startswith("cuda"):
        torch.cuda.synchronize()


def _fmt_fixed(x: float | None, nd: int = DECIMALS) -> str:
    """Plain decimal string; no scientific notation."""
    if x is None:
        return "n/a"
    if isinstance(x, float) and (math.isnan(x) or math.isinf(x)):
        return str(x)
    xf = float(x)
    return f"{xf:.{nd}f}"


def _fmt_int(x: int | None) -> str:
    if x is None:
        return "n/a"
    return str(int(x))


def _round_scalar(x: Any, nd: int = DECIMALS) -> Any:
    if isinstance(x, float):
        if math.isnan(x) or math.isinf(x):
            return x
        return round(x, nd)
    return x


def _round_for_yaml(obj: Any, nd: int = DECIMALS) -> Any:
    if isinstance(obj, dict):
        return {k: _round_for_yaml(v, nd) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_round_for_yaml(v, nd) for v in obj]
    if isinstance(obj, float):
        return _round_scalar(obj, nd)
    return obj


def _table(title: str, rows: list[tuple[str, str]], widths: tuple[int, int] | None = None) -> list[str]:
    """ASCII table: column0 = label, column1 = value."""
    w0, w1 = widths if widths else (42, 58)
    sep = "+" + "-" * (w0 + 2) + "+" + "-" * (w1 + 2) + "+"
    out = ["", title, sep]
    for a, b in rows:
        out.append(f"| {str(a)[:w0]:<{w0}} | {str(b)[:w1]:<{w1}} |")
    out.append(sep)
    return out


def _table_multi(title: str, header: list[str], data_rows: list[list[str]], col_widths: list[int]) -> list[str]:
    sep = "+" + "".join("-" * (w + 2) + "+" for w in col_widths)
    lines = ["", title, sep]
    head = "|" + "".join(f" {header[i][: col_widths[i]]:<{col_widths[i]}} |" for i in range(len(header)))
    lines.append(head)
    lines.append(sep)
    for row in data_rows:
        line = "|"
        for i, cell in enumerate(row):
            w = col_widths[i]
            cell_s = str(cell)[:w]
            line += f" {cell_s:<{w}} |"
        lines.append(line)
    lines.append(sep)
    return lines


def _transpose_summary_table(header: list[str], data_rows: list[list[str]]) -> tuple[list[str], list[list[str]]]:
    """Was one row per split; becomes one row per metric and one column per split."""
    if not data_rows:
        return [], []
    split_names = [row[0] for row in data_rows]
    new_header = ["metric"] + split_names
    new_rows: list[list[str]] = []
    for j in range(1, len(header)):
        new_rows.append([header[j]] + [data_rows[i][j] for i in range(len(data_rows))])
    return new_header, new_rows


def _table_col_widths(header: list[str], rows: list[list[str]]) -> list[int]:
    n = len(header)
    w = [len(str(header[i])) for i in range(n)]
    for row in rows:
        for i, cell in enumerate(row):
            if i < n:
                w[i] = max(w[i], len(str(cell)))
    return w


def _table5_leading_m_global_local(mc) -> tuple[float, float]:
    D, Dl = mc.d_model, mc.d_local
    Lg, Ll = mc.n_layers, mc.n_local_layers
    e_ff = mc.d_ff_mult
    V = mc.vocab_size
    m_global = Lg * (4 * D * D + 2 * e_ff * D * D)
    m_local = Ll * (4 * Dl * Dl + 2 * e_ff * Dl * Dl) + Dl * V
    return float(m_global), float(m_local)


def _spacebyte_theoretical_flops_parts(mc, use_global_blocks: bool) -> dict | None:
    if not use_global_blocks:
        _, m_local = _table5_leading_m_global_local(mc)
        Ll, W, Dl = mc.n_local_layers, mc.local_attention_window, mc.d_local
        local_pt = 2.0 * m_local + 2.0 * Ll * (2.0 * W * Dl)
        return {
            "global_pt": 0.0,
            "local_pt": float(local_pt),
            "nominal_total_pt": float(local_pt),
            "g1": 0.0,
            "g2": 0.0,
        }

    T_loc = mc.context_size
    TG = mc.global_context_size
    W = mc.local_attention_window
    D, Dl = mc.d_model, mc.d_local
    Lg, Ll = mc.n_layers, mc.n_local_layers
    m_global, m_local = _table5_leading_m_global_local(mc)
    r = TG / T_loc
    g1 = 2.0 * m_global * r
    g2 = 2.0 * Lg * (2.0 * TG * D) * r
    global_pt = g1 + g2
    local_pt = 2.0 * m_local + 2.0 * Ll * (2.0 * W * Dl)
    return {
        "global_pt": float(global_pt),
        "local_pt": float(local_pt),
        "nominal_total_pt": float(global_pt + local_pt),
        "g1": float(g1),
        "g2": float(g2),
    }


def _adjusted_flops_per_position(flops_parts: dict, global_util: float) -> float:
    u = max(0.0, min(1.0, global_util))
    return flops_parts["local_pt"] + u * flops_parts["global_pt"]


def _code_accounting_flops_per_token(model, mc) -> float | None:
    if not hasattr(model, "n_mult_add"):
        return None
    total = model.n_flops(training=False)
    return float(total) / float(mc.context_size)


def _flops_per_byte(flops_per_position: float, bytes_per_token: float) -> float:
    bt = float(bytes_per_token)
    return flops_per_position / bt if bt > 0 else flops_per_position


def _fmt_scalar(v):
    if isinstance(v, (np.ndarray, torch.Tensor)):
        v = np.asarray(v).flat[0]
    return float(v)


def prepare_model_config(Model, raw_cfg: dict, overrides: dict) -> dict:
    cfg = dict(raw_cfg)
    for k, v in overrides.items():
        if hasattr(Model.Config, k):
            cfg[k] = v
    return cfg


def build_model(Model, cfg: dict, state_dict: dict, device: str):
    model_config = Model.Config(**cfg)
    model = Model(model_config).to(device)
    model.load_state_dict(state_dict, strict=True)
    model.eval()
    return model, model_config


def resolve_checkpoint_path(checkpoint: str, checkpoint_file: str | None) -> str:
    p = Path(checkpoint)
    if p.is_dir():
        name = checkpoint_file or "ckpt.pt"
        return str(p / name)
    return str(p)


def sample_line_snippets(
    dataset_name: str,
    split: str,
    n: int,
    seed: int,
    max_chars: int = 500,
) -> list[dict[str, str]]:
    fn = _SPLIT_FILES.get(split)
    if fn is None:
        return []
    path = _ROOT / "datasets" / dataset_name / fn
    if not path.is_file():
        return []
    size = path.stat().st_size
    if size < 2:
        return []
    rng = random.Random(seed)
    out: list[dict[str, str]] = []
    with open(path, "rb") as f:
        for _ in range(n * 4):
            if len(out) >= n:
                break
            start = rng.randint(0, max(0, size - 2))
            f.seek(start)
            f.readline()
            raw = f.readline()
            if not raw.strip():
                continue
            text = raw.decode("utf-8", errors="replace").strip()
            if len(text) > max_chars:
                text = text[:max_chars] + "…"
            out.append({"split": split, "text": text})
    return out[:n]


def sample_examples_across_splits(
    dataset_name: str,
    split_order: list[str],
    n_total: int,
    seed: int,
    max_chars: int = 500,
) -> list[dict[str, str]]:
    """Sample random corpus lines, splitting ``n_total`` across splits in ``split_order`` (order preserved)."""
    valid = [
        sp
        for sp in split_order
        if sp in _SPLIT_FILES and (_ROOT / "datasets" / dataset_name / _SPLIT_FILES[sp]).is_file()
    ]
    if not valid or n_total <= 0:
        return []
    k = len(valid)
    base = n_total // k
    rem = n_total % k
    out: list[dict[str, str]] = []
    for i, sp in enumerate(valid):
        n_here = base + (1 if i < rem else 0)
        if n_here <= 0:
            continue
        part = sample_line_snippets(
            dataset_name, sp, n_here, seed + i * 100_003, max_chars=max_chars
        )
        out.extend(part)
    return out[:n_total]


def _preview_line(s: str, max_chars: int) -> str:
    t = s.replace("\n", "↵ ")
    return t if len(t) <= max_chars else t[: max_chars - 1] + "…"


def _truncate_prompt_tokens(tok: torch.Tensor, max_len: int) -> torch.Tensor:
    """Keep BOS (index 0) and the tail of the prompt so length ≤ max_len."""
    max_len = max(1, max_len)
    if tok.numel() <= max_len:
        return tok
    bos = tok[:1]
    if max_len == 1:
        return bos
    tail = tok[1:][-(max_len - 1) :]
    return torch.cat([bos, tail], dim=0)


def generate_examples_across_splits(
    model: torch.nn.Module,
    tokenizer: util.Tokenizer,
    dataset_name: str,
    split_order: list[str],
    n_total: int,
    seed: int,
    *,
    device: str,
    context_size: int,
    gen_tokens: int,
    prompt_max_chars: int,
    temperature: float,
    top_k: int | None,
    autocast_ctx,
    use_tqdm: bool = True,
) -> list[dict[str, str]]:
    """Sample prompts from corpus per split, then continue with ``model.generate``."""
    prompts = sample_examples_across_splits(
        dataset_name, split_order, n_total, seed, max_chars=prompt_max_chars
    )
    max_total = context_size + 1
    gen_tokens = max(1, min(int(gen_tokens), max_total - 1))
    out: list[dict[str, str]] = []

    sample_iter = prompts
    if use_tqdm:
        sample_iter = tqdm(prompts, desc="Generating samples...", leave=False)
    else:
        print("Generating samples...", flush=True)

    for idx, item in enumerate(sample_iter):
        line = item["text"]
        if line.endswith("…"):
            line = line[:-1]
        line = line[:prompt_max_chars]
        sp = item["split"]
        tok = tokenizer.encode(line, prepend_BOS=True, device=device)
        max_prompt_len = max(1, max_total - gen_tokens)
        tok = _truncate_prompt_tokens(tok, max_prompt_len)
        lp = int(tok.numel())
        max_tokens = min(max_total, lp + gen_tokens)
        start = tok.unsqueeze(0)
        torch.manual_seed(seed + idx * 100_019)
        with torch.inference_mode():
            with autocast_ctx:
                full, _ = model.generate(
                    start,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    top_k=top_k,
                )
        prompt_txt = tokenizer.decode(tok[1:])
        gen_txt = tokenizer.decode(full[0, lp:])
        out.append({"split": sp, "prompt": prompt_txt, "generated": gen_txt})
    return out


def main():
    parser = argparse.ArgumentParser(description="Validate checkpoints; tabular output and optional YAML.")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--checkpoint-file", type=str, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--eval-iters", type=int, default=1000000)
    parser.add_argument("--splits", type=str, default="train,val,test")
    parser.add_argument("--local-attention-window", type=int, default=None)
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--use-global-blocks", dest="use_global_blocks", action="store_true", default=None)
    group.add_argument("--no-global-blocks", dest="use_global_blocks", action="store_false")
    parser.add_argument("--no-tqdm", action="store_true")
    parser.add_argument(
        "--examples",
        type=int,
        default=10,
        help="Number of model-generated continuations (0 to disable); prompts sampled from corpus across --splits.",
    )
    parser.add_argument("--examples-seed", type=int, default=42)
    parser.add_argument(
        "--examples-gen-tokens",
        type=int,
        default=128,
        help="New tokens to sample per example (total length is capped by model context_size).",
    )
    parser.add_argument(
        "--examples-prompt-chars",
        type=int,
        default=512,
        help="Max characters per corpus line before tokenization (prompt may be truncated further by context).",
    )
    parser.add_argument("--examples-temperature", type=float, default=1.0)
    parser.add_argument("--examples-top-k", type=int, default=None, help="Optional top-k sampling for generation.")
    parser.add_argument("--output-yaml", type=str, default=None, help="Write full report to this YAML file.")
    args = parser.parse_args()

    if args.eval_iters < 3:
        print(
            "warning: eval_iters < 3 breaks variance stats in estimate_loss; using 3.",
            file=sys.stderr,
        )
        args.eval_iters = 3

    os.chdir(_ROOT)

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    ckpt_path = resolve_checkpoint_path(args.checkpoint, args.checkpoint_file)
    if not os.path.isfile(ckpt_path):
        print(f"error: checkpoint file not found: {ckpt_path}", file=sys.stderr)
        sys.exit(1)

    import megabyte  # noqa: F401
    import transformer  # noqa: F401
    from megabyte import MegaByte  # noqa: F401
    from spacebyte import SpaceByte, SpaceByteConfig  # noqa: F401
    from transformer import Transformer  # noqa: F401

    from train import estimate_loss
    import data as data_mod

    checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
    Model = eval(checkpoint["model"])
    raw_cfg = dict(checkpoint["model_config"])
    for k in list(raw_cfg):
        if not hasattr(Model.Config, k):
            del raw_cfg[k]
    state_dict = checkpoint["state_dict"]

    overrides = {}
    if args.local_attention_window is not None:
        overrides["local_attention_window"] = args.local_attention_window
    if args.use_global_blocks is not None:
        overrides["use_global_blocks"] = args.use_global_blocks

    cfg = prepare_model_config(Model, raw_cfg, overrides)
    model, mc = build_model(Model, cfg, state_dict, device)

    flops_parts = None
    m_global_l = m_local_l = None
    if isinstance(mc, SpaceByteConfig):
        m_global_l, m_local_l = _table5_leading_m_global_local(mc)
        flops_parts = _spacebyte_theoretical_flops_parts(mc, getattr(mc, "use_global_blocks", True))

    code_flops_per_token = _code_accounting_flops_per_token(model, mc)

    tc = checkpoint["train_config"]
    d = data_mod.dataset(tc["dataset"], mc.tokenizer)
    model.dataset_tokenizer = d.tokenizer

    bytes_per_tok = float(d.bytes_per_token)
    total_tokens_ckpt = checkpoint.get("total_tokens")
    trained_bytes = float(total_tokens_ckpt) * bytes_per_tok if total_tokens_ckpt is not None else None
    n_params_ne = model.num_params(embedding=False) if hasattr(model, "num_params") else None
    trained_over_ne = trained_bytes / n_params_ne if trained_bytes is not None and n_params_ne else None

    mB = tc["micro_batch_size"]
    autocast = util.autocast_context(tc["dtype"])
    data_seed = tc["data_seed"]

    splits = [s.strip() for s in args.splits.split(",") if s.strip()]
    example_splits = [s for s in splits if s in d.splits()]

    report: dict[str, Any] = {
        "meta": {
            "checkpoint": ckpt_path,
            "device": device,
            "dtype": tc["dtype"],
            "eval_iters": args.eval_iters,
            "micro_batch_size": mB,
            "splits_evaluated": splits,
            "examples_splits": example_splits,
            "examples_generation": {
                "gen_tokens": args.examples_gen_tokens,
                "prompt_max_chars": args.examples_prompt_chars,
                "temperature": args.examples_temperature,
                "top_k": args.examples_top_k,
            },
            "dataset_bytes_per_token": round(bytes_per_tok, DECIMALS),
        },
        "model": {
            "local_attention_window": mc.local_attention_window,
            "use_global_blocks": getattr(mc, "use_global_blocks", True),
            "context_size": mc.context_size,
            "global_context_size": getattr(mc, "global_context_size", None),
        },
        "training_checkpoint": {},
        "theoretical_flops": {},
        "per_split": {},
        "examples": [],
    }

    report["training_checkpoint"]["total_tokens"] = int(total_tokens_ckpt) if total_tokens_ckpt is not None else None
    report["training_checkpoint"]["trained_bytes"] = round(trained_bytes, DECIMALS) if trained_bytes else None
    report["training_checkpoint"]["non_embedding_parameters"] = int(n_params_ne) if n_params_ne else None
    report["training_checkpoint"]["trained_bytes_per_parameter"] = (
        round(trained_over_ne, DECIMALS) if trained_over_ne is not None else None
    )

    if flops_parts is not None:
        report["theoretical_flops"]["param_counts_leading_m_global"] = int(m_global_l) if m_global_l else None
        report["theoretical_flops"]["param_counts_leading_m_local"] = int(m_local_l) if m_local_l else None
        report["theoretical_flops"]["global_pathway_flops_per_byte"] = round(
            _flops_per_byte(flops_parts["global_pt"], bytes_per_tok), DECIMALS
        )
        report["theoretical_flops"]["local_pathway_flops_per_byte"] = round(
            _flops_per_byte(flops_parts["local_pt"], bytes_per_tok), DECIMALS
        )
        report["theoretical_flops"]["nominal_total_flops_per_byte"] = round(
            _flops_per_byte(flops_parts["nominal_total_pt"], bytes_per_tok), DECIMALS
        )
        report["theoretical_flops"]["global_layer_forwards_per_micro_batch"] = mc.n_layers
        report["theoretical_flops"]["global_layer_forwards_per_split"] = args.eval_iters * mc.n_layers

    if code_flops_per_token is not None:
        report["theoretical_flops"]["code_accounting_flops_per_byte"] = round(
            _flops_per_byte(code_flops_per_token, bytes_per_tok), DECIMALS
        )

    lines_out: list[str] = []

    lines_out += _table(
        "Run configuration",
        [
            ("checkpoint", ckpt_path),
            ("device / dtype", f"{device} / {tc['dtype']}"),
            ("eval_iters", _fmt_int(args.eval_iters)),
            ("micro_batch_size", _fmt_int(mB)),
            ("splits", ", ".join(splits)),
            ("dataset bytes_per_token", _fmt_fixed(bytes_per_tok)),
        ],
    )

    lines_out += _table(
        "Training data vs parameters",
        [
            ("checkpoint total_tokens", _fmt_int(int(total_tokens_ckpt)) if total_tokens_ckpt else "n/a"),
            ("trained_bytes", _fmt_fixed(trained_bytes, 0) if trained_bytes else "n/a"),
            ("non_embedding_parameters", _fmt_int(int(n_params_ne)) if n_params_ne else "n/a"),
            ("trained_bytes / non_embedding_parameters", _fmt_fixed(trained_over_ne) if trained_over_ne else "n/a"),
        ],
    )

    if flops_parts is not None:
        g_pb = _flops_per_byte(flops_parts["global_pt"], bytes_per_tok)
        l_pb = _flops_per_byte(flops_parts["local_pt"], bytes_per_tok)
        n_pb = _flops_per_byte(flops_parts["nominal_total_pt"], bytes_per_tok)
        ug = getattr(mc, "use_global_blocks", True)
        flop_rows = [
            ("leading param count m_global", _fmt_int(int(m_global_l)) if m_global_l else "n/a"),
            ("leading param count m_local", _fmt_int(int(m_local_l)) if m_local_l else "n/a"),
            ("global pathway FLOPs/byte", _fmt_fixed(g_pb)),
            ("local pathway FLOPs/byte", _fmt_fixed(l_pb)),
            ("nominal total FLOPs/byte", _fmt_fixed(n_pb)),
        ]
        if ug:
            flop_rows.append(("global TransformerBlock forwards per micro-batch", _fmt_int(mc.n_layers)))
            flop_rows.append(
                ("global layer-forwards per split (eval_iters × n_layers)", _fmt_int(args.eval_iters * mc.n_layers))
            )
        lines_out += _table("Theoretical inference FLOPs (SpaceByte-style decomposition)", flop_rows)

    if code_flops_per_token is not None:
        c_pb = _flops_per_byte(code_flops_per_token, bytes_per_tok)
        lines_out += _table(
            "Code accounting (n_flops / context_size)",
            [
                ("FLOPs/byte", _fmt_fixed(c_pb)),
            ],
        )
        if isinstance(mc, SpaceByteConfig) and not getattr(mc, "use_global_blocks", True):
            lines_out.append(
                "(forward skips global blocks; n_mult_add may still count global stack — code bound loose.)"
            )

    summary_rows_header = [
        "split",
        "CE (NAT)",
        "CE (BPB)",
        "PPL",
        "Wall time (s)",
        "logits/s",
        "micro-batches/s",
        "Global %",
        "max FLOPs/B",
        "FLOPs/B",
        "max FLOPs/s",
        "FLOPs/s",
        "real FLOPs/B",
    ]
    summary_data: list[list[str]] = []

    for split in splits:
        if split not in d.splits():
            print(f"warning: split '{split}' not in dataset; available: {list(d.splits())}", file=sys.stderr)
            continue
        it = d.iter(split, context_size=mc.context_size, batch_size=mB, seed=data_seed, device=device)
        _sync_cuda_if_needed(device)
        t_wall0 = time.perf_counter()
        losses = estimate_loss(
            it,
            args.eval_iters,
            model,
            bytes_per_token=d.bytes_per_token,
            autocast=autocast,
            desc=None if args.no_tqdm else split,
        )
        _sync_cuda_if_needed(device)
        wall_s = time.perf_counter() - t_wall0

        tokens_total = args.eval_iters * mB * mc.context_size
        thr_tokens_s = tokens_total / wall_s if wall_s > 0 else float("nan")
        thr_mb_s = args.eval_iters / wall_s if wall_s > 0 else float("nan")

        gc_mean = None
        pct_global = None
        if (
            isinstance(mc, SpaceByteConfig)
            and getattr(mc, "use_global_blocks", True)
            and flops_parts is not None
            and "global context" in losses
        ):
            gc_mean = _fmt_scalar(losses["global context"])
            TGn = mc.global_context_size
            pct_global = gc_mean / TGn if TGn else float("nan")

        ce = _fmt_scalar(losses.get("cross entropy") or losses.get("loss"))
        bpb = _fmt_scalar(losses["bits per byte"]) if "bits per byte" in losses else None
        ppl = math.exp(ce)

        tot_nom_flops = flops_parts["nominal_total_pt"] * tokens_total if flops_parts else None
        fl_nom_s = tot_nom_flops / wall_s if wall_s > 0 and tot_nom_flops else float("nan")

        adj_total_flops = None
        adj_pt = None
        fl_adj_s = float("nan")
        a_pb = float("nan")
        if pct_global is not None and flops_parts is not None:
            adj_pt = _adjusted_flops_per_position(flops_parts, pct_global)
            adj_total_flops = adj_pt * tokens_total
            fl_adj_s = adj_total_flops / wall_s if wall_s > 0 else float("nan")
            a_pb = _flops_per_byte(adj_pt, bytes_per_tok)

        n_pb_nom = (
            _flops_per_byte(flops_parts["nominal_total_pt"], bytes_per_tok) if flops_parts else None
        )
        c_pb_code = _flops_per_byte(code_flops_per_token, bytes_per_tok) if code_flops_per_token else None

        ps = {
            "cross_entropy": round(ce, DECIMALS),
            "cross_entropy_bpb": round(bpb, DECIMALS) if bpb is not None else None,
            "perplexity": round(ppl, DECIMALS),
            "wall_seconds": round(wall_s, DECIMALS),
            "predicted_positions_total": int(tokens_total),
            "raw_bytes_total": round(tokens_total * bytes_per_tok, DECIMALS),
            "predicted_positions_per_second": round(thr_tokens_s, DECIMALS),
            "micro_batches_per_second": round(thr_mb_s, DECIMALS),
            "mean_global_T": round(gc_mean, DECIMALS) if gc_mean is not None else None,
            "global_patch_utilization_fraction": round(pct_global, DECIMALS) if pct_global is not None else None,
            "global_patch_utilization_percent": round(100 * pct_global, DECIMALS) if pct_global is not None else None,
            "nominal_flops_per_byte": round(n_pb_nom, DECIMALS) if n_pb_nom is not None else None,
            "global_patch_util_adjusted_flops_per_byte": round(a_pb, DECIMALS)
            if pct_global is not None and flops_parts
            else None,
            "nominal_flops_per_second": round(fl_nom_s, DECIMALS) if tot_nom_flops else None,
            "global_patch_util_adjusted_flops_per_second": round(fl_adj_s, DECIMALS) if adj_total_flops else None,
            "code_flops_per_byte": round(c_pb_code, DECIMALS) if c_pb_code is not None else None,
        }
        report["per_split"][split] = ps

        summary_data.append(
            [
                split,
                _fmt_fixed(ce),
                _fmt_fixed(bpb) if bpb is not None else "n/a",
                _fmt_fixed(ppl),
                _fmt_fixed(wall_s),
                _fmt_fixed(thr_tokens_s),
                _fmt_fixed(thr_mb_s),
                _fmt_fixed(100 * pct_global) if pct_global is not None else "n/a",
                _fmt_fixed(n_pb_nom) if n_pb_nom is not None else "n/a",
                _fmt_fixed(a_pb) if pct_global is not None and flops_parts else "n/a",
                _fmt_fixed(fl_nom_s) if tot_nom_flops else "n/a",
                _fmt_fixed(fl_adj_s) if adj_total_flops else "n/a",
                _fmt_fixed(c_pb_code) if c_pb_code is not None else "n/a",
            ]
        )

        lines_out.append("")
        lines_out.append(f"Split «{split}» (eval_iters={args.eval_iters}) — detailed losses")
        loss_lines = []
        for name in sorted(losses.keys()):
            if name.endswith(" stat"):
                continue
            disp = name
            if name == "bits per byte":
                disp = "Cross Entropy (BPB)"
            v = losses[name]
            err = losses.get(name + " stat")
            if err is not None and isinstance(err, (float, np.floating, int, np.integer)):
                loss_lines.append(
                    (disp, f"{_fmt_fixed(_fmt_scalar(v))} ± {_fmt_fixed(_fmt_scalar(err))}")
                )
            else:
                av = np.asarray(v)
                if av.shape == ():
                    loss_lines.append((disp, _fmt_fixed(float(av))))
                else:
                    loss_lines.append((disp, "<see checkpoint metrics; tensor omitted>"))
        lines_out += _table(f"Metrics ({split})", loss_lines)

    if summary_data:
        sum_header, sum_rows = _transpose_summary_table(summary_rows_header, summary_data)
        lines_out += _table_multi(
            "Summary — all splits",
            sum_header,
            sum_rows,
            _table_col_widths(sum_header, sum_rows),
        )
        lines_out.append(
            "Legend: CE (NAT) = cross-entropy (nats); CE (BPB) = Cross Entropy in bits/byte; "
            "Global % = 100×mean(global_T)/global_context_size; "
            "max FLOPs/B = nominal theoretical SpaceByte FLOPs per raw byte; "
            "FLOPs/B = utilization-adjusted FLOPs per raw byte (local + Global%×global pathway); "
            "max FLOPs/s = nominal FLOPs per second; FLOPs/s = adjusted FLOPs per second; "
            "real FLOPs/B = code-accounting FLOPs per raw byte (n_flops/context_size); "
            "logits/s = micro_batch_size×context_size / wall_time_per_micro_batch step."
        )

    if args.examples > 0:
        if not example_splits:
            lines_out.append("")
            lines_out.append("Examples: none of --splits are present for this dataset; skipped.")
        else:
            tokzr = util.Tokenizer(mc.tokenizer)
            ex = generate_examples_across_splits(
                model,
                tokzr,
                tc["dataset"],
                example_splits,
                args.examples,
                args.examples_seed,
                device=device,
                context_size=mc.context_size,
                gen_tokens=args.examples_gen_tokens,
                prompt_max_chars=args.examples_prompt_chars,
                temperature=args.examples_temperature,
                top_k=args.examples_top_k,
                autocast_ctx=autocast,
                use_tqdm=not args.no_tqdm,
            )
            report["examples"] = ex
            split_desc = ", ".join(example_splits)
            lines_out.append("")
            lines_out.append(
                f"Generated text (prompt from corpus → continuation), splits: {split_desc}, n={len(ex)}"
            )
            for i, item in enumerate(ex, 1):
                pr = _preview_line(item["prompt"], 180)
                ge = _preview_line(item["generated"], 400)
                lines_out.append(f"  [{i}] ({item['split']}) prompt: {pr}")
                lines_out.append(f"       → {ge}")

    text_report = "\n".join(lines_out)
    print(text_report)

    if args.output_yaml:
        try:
            import yaml
        except ImportError:
            print("error: PyYAML required for --output-yaml (pip install pyyaml)", file=sys.stderr)
            sys.exit(1)
        out_path = Path(args.output_yaml)
        payload = _round_for_yaml(report)
        with open(out_path, "w", encoding="utf-8") as yf:
            yaml.safe_dump(
                payload,
                yf,
                default_flow_style=False,
                allow_unicode=True,
                sort_keys=False,
                width=120,
            )
        print(f"\nWrote YAML report to {out_path.resolve()}")


if __name__ == "__main__":
    main()
