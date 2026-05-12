#!/usr/bin/env python3
"""
Validate SpaceByte with sequence-level adaptive routing.

For each micro-batch item the script first chooses routing parameters for the
last byte in the context, then runs SpaceByte with those parameters:

* router mode: use a trained router checkpoint for ``use_global_blocks`` and
  ``local_attention_window``;
* baseline mode: match standard SpaceByte by running global blocks for the whole
  forward (SpaceByte still uses UTF-8 internally to choose ``global_ts``) and
  using the checkpoint's default local attention window.

With ``--best-effort``, sweeps routing flags (with/without ``--no-global-routing``
and ``--no-window-routing``) and classification thresholds 0.0–1.0 (step 0.1),
writes ``best_effort_results.csv`` / ``best_effort_table.md``, and saves per-mode
comparison plots (baseline vs method: BPB and real FLOPs/B) under
``--best-effort-output-dir``.
"""

from __future__ import annotations

import argparse
import collections
import contextlib
import dataclasses
import math
import os
import re
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import csv

import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoTokenizer

import util
from validate import (
    DECIMALS,
    _adjusted_flops_per_position,
    _code_accounting_flops_per_token,
    _flops_per_byte,
    _fmt_fixed,
    _fmt_int,
    _fmt_scalar,
    _round_for_yaml,
    _spacebyte_theoretical_flops_parts,
    _sync_cuda_if_needed,
    _table,
    _table5_leading_m_global_local,
    _table_col_widths,
    _table_multi,
    _transpose_summary_table,
    build_model,
    prepare_model_config,
    resolve_checkpoint_path,
)

_ROOT = Path(__file__).resolve().parent

ROUTING_MODES_BEST_EFFORT: tuple[tuple[bool, bool, str], ...] = (
    (False, False, "full_router"),
    (False, True, "router_global_only"),
    (True, False, "router_window_only"),
    (True, True, "spacebyte_default"),
)


@dataclasses.dataclass(frozen=True)
class RouterBundle:
    router: torch.nn.Module
    hf_tokenizer: Any
    max_length: int
    router_device: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate SpaceByte with a router-controlled adaptive compute policy."
    )
    parser.add_argument("--checkpoint", type=str, required=True, help="SpaceByte checkpoint or run dir.")
    parser.add_argument("--checkpoint-file", type=str, default=None)
    parser.add_argument(
        "--router-checkpoint",
        type=str,
        default=None,
        help="Router checkpoint, required unless --baseline is set.",
    )
    parser.add_argument(
        "--classification-threshold",
        type=float,
        default=0.5,
        help="Probability threshold for router P(use_global_blocks=1).",
    )
    parser.add_argument(
        "--baseline",
        action="store_true",
        help="Use standard SpaceByte: global blocks enabled and default local attention window.",
    )
    parser.add_argument(
        "--no-global-routing",
        action="store_true",
        help="Do not use router classification; keep SpaceByte global blocks enabled.",
    )
    parser.add_argument(
        "--no-window-routing",
        action="store_true",
        help="Use the checkpoint default local attention window instead of router regression.",
    )
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--use-global-blocks",
        "--with-global-blocks",
        "--with-global-block",
        dest="use_global_blocks",
        action="store_true",
        default=None,
        help="Force global blocks on for every sequence, bypassing router classification.",
    )
    group.add_argument(
        "--no-global-blocks",
        dest="use_global_blocks",
        action="store_false",
        help="Force global blocks off for every sequence, bypassing router classification.",
    )
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--router-device", type=str, default=None)
    parser.add_argument("--eval-iters", type=int, default=1000000)
    parser.add_argument("--splits", type=str, default="train,val,test")
    parser.add_argument("--no-tqdm", action="store_true")
    parser.add_argument("--output-yaml", type=str, default=None, help="Write full report to YAML.")
    parser.add_argument(
        "--best-effort",
        action="store_true",
        help="Grid over routing flags and classification thresholds; write table + comparison plots.",
    )
    parser.add_argument(
        "--best-effort-output-dir",
        type=str,
        default="validate_sota_best_effort",
        help="Output directory for --best-effort (CSV, Markdown, PNGs).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if getattr(args, "best_effort", False):
        run_best_effort(args)
        return
    if args.baseline:
        args.no_global_routing = True
        args.no_window_routing = True
        args.use_global_blocks = True
    uses_router_global = not args.no_global_routing and args.use_global_blocks is None
    uses_router_window = not args.no_window_routing
    needs_router = uses_router_global or uses_router_window
    if needs_router and args.router_checkpoint is None:
        raise SystemExit("error: --router-checkpoint is required unless --baseline is set")
    if args.eval_iters < 3:
        print(
            "warning: eval_iters < 3 breaks variance stats in estimate_loss; using 3.",
            file=sys.stderr,
        )
        args.eval_iters = 3

    os.chdir(_ROOT)
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    import data as data_mod
    import megabyte  # noqa: F401
    import transformer  # noqa: F401
    from megabyte import MegaByte  # noqa: F401
    from spacebyte import SpaceByte, SpaceByteConfig  # noqa: F401
    from transformer import Transformer  # noqa: F401

    ckpt_path = resolve_checkpoint_path(args.checkpoint, args.checkpoint_file)
    if not os.path.isfile(ckpt_path):
        raise SystemExit(f"error: checkpoint file not found: {ckpt_path}")

    checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
    Model = eval(checkpoint["model"])
    raw_cfg = dict(checkpoint["model_config"])
    for key in list(raw_cfg):
        if not hasattr(Model.Config, key):
            del raw_cfg[key]
    cfg = prepare_model_config(Model, raw_cfg, {})
    model, mc = build_model(Model, cfg, checkpoint["state_dict"], device)
    if not isinstance(mc, SpaceByteConfig):
        raise SystemExit("error: validate_sota.py currently supports SpaceByte checkpoints only")

    tc = checkpoint["train_config"]
    d = data_mod.dataset(tc["dataset"], mc.tokenizer)
    model.dataset_tokenizer = d.tokenizer

    router_policy = build_policy(args, d.tokenizer, int(mc.local_attention_window), device)

    bytes_per_tok = float(d.bytes_per_token)
    total_tokens_ckpt = checkpoint.get("total_tokens")
    trained_bytes = float(total_tokens_ckpt) * bytes_per_tok if total_tokens_ckpt is not None else None
    n_params_ne = model.num_params(embedding=False) if hasattr(model, "num_params") else None
    trained_over_ne = trained_bytes / n_params_ne if trained_bytes is not None and n_params_ne else None

    mB = int(tc["micro_batch_size"])
    autocast = util.autocast_context(tc["dtype"])
    data_seed = int(tc["data_seed"])
    splits = [s.strip() for s in args.splits.split(",") if s.strip()]

    m_global_l, m_local_l = _table5_leading_m_global_local(mc)
    flops_parts = _spacebyte_theoretical_flops_parts(mc, True)
    code_flops_per_token = _code_accounting_flops_per_token(model, mc)

    report: dict[str, Any] = {
        "meta": {
            "checkpoint": ckpt_path,
            "router_checkpoint": args.router_checkpoint if needs_router else None,
            "baseline": bool(args.baseline),
            "no_global_routing": bool(args.no_global_routing),
            "no_window_routing": bool(args.no_window_routing),
            "use_global_blocks_override": args.use_global_blocks,
            "classification_threshold": float(args.classification_threshold),
            "device": device,
            "router_device": router_policy.device,
            "dtype": tc["dtype"],
            "eval_iters": args.eval_iters,
            "micro_batch_size": mB,
            "splits_evaluated": splits,
            "dataset_bytes_per_token": round(bytes_per_tok, DECIMALS),
        },
        "model": {
            "default_local_attention_window": mc.local_attention_window,
            "context_size": mc.context_size,
            "global_context_size": mc.global_context_size,
        },
        "training_checkpoint": {},
        "theoretical_flops": {},
        "per_split": {},
    }
    report["training_checkpoint"]["total_tokens"] = int(total_tokens_ckpt) if total_tokens_ckpt is not None else None
    report["training_checkpoint"]["trained_bytes"] = round(trained_bytes, DECIMALS) if trained_bytes else None
    report["training_checkpoint"]["non_embedding_parameters"] = int(n_params_ne) if n_params_ne else None
    report["training_checkpoint"]["trained_bytes_per_parameter"] = (
        round(trained_over_ne, DECIMALS) if trained_over_ne is not None else None
    )
    report["theoretical_flops"]["param_counts_leading_m_global"] = int(m_global_l)
    report["theoretical_flops"]["param_counts_leading_m_local"] = int(m_local_l)
    report["theoretical_flops"]["global_pathway_flops_per_byte"] = round(
        _flops_per_byte(flops_parts["global_pt"], bytes_per_tok), DECIMALS
    )
    report["theoretical_flops"]["local_pathway_flops_per_byte_default_window"] = round(
        _flops_per_byte(flops_parts["local_pt"], bytes_per_tok), DECIMALS
    )
    if code_flops_per_token is not None:
        report["theoretical_flops"]["code_accounting_flops_per_byte_default_window"] = round(
            _flops_per_byte(code_flops_per_token, bytes_per_tok), DECIMALS
        )

    lines_out: list[str] = []
    lines_out += _table(
        "Run configuration",
        [
            ("checkpoint", ckpt_path),
            ("mode", routing_mode_name(args)),
            ("router checkpoint", "n/a" if not needs_router else str(args.router_checkpoint)),
            ("global routing", describe_global_routing(args)),
            ("window routing", "default local attention window" if args.no_window_routing else "router regressor"),
            ("classification threshold", _fmt_fixed(args.classification_threshold)),
            ("device / dtype", f"{device} / {tc['dtype']}"),
            ("router device", router_policy.device),
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

    summary_rows_header = [
        "split",
        "CE (NAT)",
        "CE (BPB)",
        "PPL",
        "Wall time (s)",
        "logits/s",
        "micro-batches/s",
        "Global %",
        "Route global %",
        "Mean window",
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
        losses = estimate_loss_sota(
            it,
            args.eval_iters,
            model,
            router_policy,
            default_window=int(mc.local_attention_window),
            bytes_per_token=d.bytes_per_token,
            autocast=autocast,
            desc=None if args.no_tqdm else split,
        )
        _sync_cuda_if_needed(device)
        wall_s = time.perf_counter() - t_wall0

        sm = losses_to_summary(losses, mc, wall_s, args.eval_iters, mB, bytes_per_tok, flops_parts)
        ce = sm["ce"]
        bpb = sm["bpb"]
        ppl = sm["ppl"]
        thr_tokens_s = sm["thr_tokens_s"]
        thr_mb_s = sm["thr_mb_s"]
        pct_global = sm["pct_global"]
        route_global_rate = sm["route_global_rate"]
        mean_window = sm["mean_window"]
        n_pb_nom = sm["n_pb_nom"]
        a_pb = sm["a_pb"]
        fl_nom_s = sm["fl_nom_s"]
        fl_adj_s = sm["fl_adj_s"]
        c_pb_code = sm["c_pb_code"]
        tokens_total = args.eval_iters * mB * mc.context_size
        active_gc_mean = _fmt_scalar(losses.get("active global context", 0.0))

        ps = {
            "cross_entropy": round(ce, DECIMALS),
            "cross_entropy_bpb": round(bpb, DECIMALS) if bpb is not None else None,
            "perplexity": round(ppl, DECIMALS),
            "wall_seconds": round(wall_s, DECIMALS),
            "predicted_positions_total": int(tokens_total),
            "raw_bytes_total": round(tokens_total * bytes_per_tok, DECIMALS),
            "predicted_positions_per_second": round(thr_tokens_s, DECIMALS),
            "micro_batches_per_second": round(thr_mb_s, DECIMALS),
            "mean_active_global_T": round(active_gc_mean, DECIMALS),
            "active_global_utilization_fraction": round(pct_global, DECIMALS),
            "active_global_utilization_percent": round(100 * pct_global, DECIMALS),
            "route_global_rate": round(route_global_rate, DECIMALS),
            "mean_local_attention_window": round(mean_window, DECIMALS),
            "nominal_flops_per_byte": round(n_pb_nom, DECIMALS),
            "adaptive_flops_per_byte": round(a_pb, DECIMALS),
            "nominal_flops_per_second": round(fl_nom_s, DECIMALS),
            "adaptive_flops_per_second": round(fl_adj_s, DECIMALS),
            "code_flops_per_byte_estimated": round(c_pb_code, DECIMALS),
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
                _fmt_fixed(100 * pct_global),
                _fmt_fixed(100 * route_global_rate),
                _fmt_fixed(mean_window),
                _fmt_fixed(n_pb_nom),
                _fmt_fixed(a_pb),
                _fmt_fixed(fl_nom_s),
                _fmt_fixed(fl_adj_s),
                _fmt_fixed(c_pb_code),
            ]
        )

        lines_out.append("")
        lines_out.append(f"Split «{split}» (eval_iters={args.eval_iters}) — detailed losses")
        loss_lines = []
        for name in sorted(losses.keys()):
            if name.endswith(" stat"):
                continue
            disp = "Cross Entropy (BPB)" if name == "bits per byte" else name
            value = losses[name]
            err = losses.get(name + " stat")
            if err is not None and isinstance(err, (float, np.floating, int, np.integer)):
                loss_lines.append((disp, f"{_fmt_fixed(_fmt_scalar(value))} ± {_fmt_fixed(_fmt_scalar(err))}"))
            else:
                av = np.asarray(value)
                if av.shape == ():
                    loss_lines.append((disp, _fmt_fixed(float(av))))
                else:
                    loss_lines.append((disp, "<tensor omitted>"))
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
            "Legend: Global % = active mean(global_T)/global_context_size, where inactive "
            "global-block routes contribute 0; Route global % = fraction of sequences routed "
            "with global blocks; FLOPs/B is adaptive estimate using mean route window and "
            "active Global %; real FLOPs/B is the same adaptive code-level estimate."
        )

    text_report = "\n".join(lines_out)
    print(text_report)

    if args.output_yaml:
        import yaml

        out_path = Path(args.output_yaml)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w", encoding="utf-8") as f:
            yaml.safe_dump(_round_for_yaml(report), f, sort_keys=False, allow_unicode=False)


def _drop_bos(tokens: torch.Tensor, bos: int) -> torch.Tensor:
    return tokens[tokens != int(bos)]


class RoutingPolicy:
    def __init__(
        self,
        *,
        mode: str,
        default_window: int,
        dataset_tokenizer: util.Tokenizer,
        device: str,
        threshold: float,
        use_router_global: bool,
        use_router_window: bool,
        global_blocks_override: bool | None,
        router: torch.nn.Module | None = None,
        hf_tokenizer: Any | None = None,
        max_length: int | None = None,
    ) -> None:
        self.mode = mode
        self.default_window = int(default_window)
        self.dataset_tokenizer = dataset_tokenizer
        self.device = device
        self.threshold = float(threshold)
        self.use_router_global = use_router_global
        self.use_router_window = use_router_window
        self.global_blocks_override = global_blocks_override
        self.router = router
        self.hf_tokenizer = hf_tokenizer
        self.max_length = max_length

    @torch.inference_mode()
    def __call__(self, tokens: torch.Tensor) -> dict[str, torch.Tensor]:
        B = tokens.shape[0]
        default_windows = torch.full((B,), self.default_window, dtype=torch.long, device=tokens.device)
        default_ratio = torch.ones(B, dtype=torch.float32, device=tokens.device)
        forced_global = self._forced_global(tokens)

        if not self.use_router_global and not self.use_router_window:
            assert forced_global is not None
            return {
                "use_global_blocks": forced_global,
                "local_attention_window": default_windows,
                "prob_global": forced_global.to(torch.float32),
                "window_ratio": default_ratio,
            }

        assert self.router is not None and self.hf_tokenizer is not None and self.max_length is not None
        texts = [
            self.dataset_tokenizer.decode(_drop_bos(tokens[b].detach().cpu(), self.dataset_tokenizer.BOS))
            for b in range(tokens.shape[0])
        ]
        enc = self.hf_tokenizer(
            texts,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        enc = {k: v.to(self.device) for k, v in enc.items()}
        outputs = self.router(
            input_ids=enc["input_ids"],
            attention_mask=enc["attention_mask"],
            token_type_ids=enc.get("token_type_ids"),
        )
        logits = outputs["global_logits"]
        if logits.dim() == 2 and logits.size(-1) == 2:
            prob_global = torch.softmax(logits, dim=-1)[:, 1]
        else:
            prob_global = torch.sigmoid(logits.squeeze(-1))
        use_global = prob_global >= self.threshold
        if forced_global is not None:
            use_global = forced_global.to(prob_global.device)
            prob_global = forced_global.to(prob_global.device, dtype=torch.float32)
        ratio = outputs["window_ratio"].detach().float().clamp(0.0, 1.0)
        windows = torch.ceil(ratio * float(self.default_window)).long().clamp(1, self.default_window)
        if not self.use_router_window:
            ratio = torch.ones_like(ratio)
            windows = torch.full_like(windows, self.default_window)
        return {
            "use_global_blocks": use_global.to(tokens.device),
            "local_attention_window": windows.to(tokens.device),
            "prob_global": prob_global.detach().float().to(tokens.device),
            "window_ratio": ratio.to(tokens.device),
        }

    def _forced_global(self, tokens: torch.Tensor) -> torch.Tensor | None:
        if self.global_blocks_override is not None:
            value = bool(self.global_blocks_override)
        elif not self.use_router_global:
            # Standard SpaceByte behavior: run global blocks for the whole forward.
            # The UTF-8 heuristic still acts inside spacebyte.py by choosing global_ts.
            value = True
        else:
            return None
        return torch.full((tokens.shape[0],), value, dtype=torch.bool, device=tokens.device)


def load_router_bundle(router_checkpoint: str, *, main_device: str, router_device: str | None) -> RouterBundle:
    from router.config import RouterTrainConfig
    from router.model import build_router_model
    from router.train import resolve_tokenizer_name

    rd = router_device or main_device
    ckpt = torch.load(router_checkpoint, map_location="cpu", weights_only=False)
    raw_cfg = ckpt["config"]
    names = {f.name for f in dataclasses.fields(RouterTrainConfig)}
    router_cfg = RouterTrainConfig(**{k: v for k, v in raw_cfg.items() if k in names})
    router_cfg.normalize_learning_rates()
    hf_tokenizer = AutoTokenizer.from_pretrained(resolve_tokenizer_name(router_cfg))
    hf_tokenizer.truncation_side = router_cfg.tokenizer_truncation_side
    router = build_router_model(router_cfg, hf_tokenizer)
    maybe_align_classifier_head(router, ckpt["model_state_dict"])
    router.load_state_dict(ckpt["model_state_dict"], strict=True)
    router.to(rd)
    router.eval()
    return RouterBundle(router=router, hf_tokenizer=hf_tokenizer, max_length=router_cfg.max_length, router_device=rd)


def build_policy(
    args: argparse.Namespace,
    dataset_tokenizer: util.Tokenizer,
    default_window: int,
    main_device: str,
    *,
    router_bundle: RouterBundle | None = None,
) -> RoutingPolicy:
    use_router_global = not args.no_global_routing and args.use_global_blocks is None
    use_router_window = not args.no_window_routing
    if not use_router_global and not use_router_window:
        return RoutingPolicy(
            mode="baseline",
            default_window=default_window,
            dataset_tokenizer=dataset_tokenizer,
            device=main_device,
            threshold=args.classification_threshold,
            use_router_global=False,
            use_router_window=False,
            global_blocks_override=args.use_global_blocks,
        )

    if router_bundle is None:
        if args.router_checkpoint is None:
            raise SystemExit("error: --router-checkpoint is required for router-driven routing")
        bundle = load_router_bundle(args.router_checkpoint, main_device=main_device, router_device=args.router_device)
    else:
        bundle = router_bundle

    return RoutingPolicy(
        mode="router",
        default_window=default_window,
        dataset_tokenizer=dataset_tokenizer,
        device=bundle.router_device,
        threshold=args.classification_threshold,
        use_router_global=use_router_global,
        use_router_window=use_router_window,
        global_blocks_override=args.use_global_blocks,
        router=bundle.router,
        hf_tokenizer=bundle.hf_tokenizer,
        max_length=bundle.max_length,
    )


def maybe_align_classifier_head(model: torch.nn.Module, state_dict: dict[str, torch.Tensor]) -> None:
    if "classifier.4.weight" not in state_dict:
        return
    w = state_dict["classifier.4.weight"]
    current = model.classifier[-1]
    if not isinstance(current, torch.nn.Linear):
        raise TypeError("Expected classifier[-1] to be Linear")
    if current.out_features == int(w.shape[0]):
        return
    model.classifier[-1] = torch.nn.Linear(int(w.shape[1]), int(w.shape[0]))


def routing_mode_name(args: argparse.Namespace) -> str:
    if (args.no_global_routing or args.use_global_blocks is not None) and args.no_window_routing:
        return "baseline"
    if args.no_global_routing or args.use_global_blocks is not None:
        return "router-window-only"
    if args.no_window_routing:
        return "router-global-only"
    return "router"


def describe_global_routing(args: argparse.Namespace) -> str:
    if args.use_global_blocks is True:
        return "forced on for every sequence"
    if args.use_global_blocks is False:
        return "forced off for every sequence"
    if args.no_global_routing:
        return "standard SpaceByte global blocks on"
    return "router classifier"


def losses_to_summary(
    losses: dict[str, Any],
    mc: Any,
    wall_s: float,
    eval_iters: int,
    mB: int,
    bytes_per_tok: float,
    flops_parts: dict[str, float],
) -> dict[str, float | None]:
    tokens_total = eval_iters * mB * mc.context_size
    active_gc_mean = _fmt_scalar(losses.get("active global context", 0.0))
    pct_global = active_gc_mean / mc.global_context_size if mc.global_context_size else float("nan")
    route_global_rate = _fmt_scalar(losses.get("route global rate", 0.0))
    mean_window = _fmt_scalar(losses.get("route local attention window", mc.local_attention_window))
    ce = _fmt_scalar(losses.get("cross entropy") or losses.get("loss"))
    bpb = _fmt_scalar(losses["bits per byte"]) if "bits per byte" in losses else None
    ppl = math.exp(ce)

    default_nom_pt = flops_parts["nominal_total_pt"]
    max_total_flops = default_nom_pt * tokens_total
    fl_nom_s = max_total_flops / wall_s if wall_s > 0 else float("nan")

    adaptive_parts = adaptive_flops_parts(mc, mean_window)
    adaptive_pt = _adjusted_flops_per_position(adaptive_parts, pct_global)
    adaptive_total_flops = adaptive_pt * tokens_total
    fl_adj_s = adaptive_total_flops / wall_s if wall_s > 0 else float("nan")
    n_pb_nom = _flops_per_byte(default_nom_pt, bytes_per_tok)
    a_pb = _flops_per_byte(adaptive_pt, bytes_per_tok)
    c_pb_code = _flops_per_byte(adaptive_pt, bytes_per_tok)

    thr_tokens_s = tokens_total / wall_s if wall_s > 0 else float("nan")
    thr_mb_s = eval_iters / wall_s if wall_s > 0 else float("nan")

    return {
        "ce": ce,
        "bpb": bpb,
        "ppl": ppl,
        "wall_s": wall_s,
        "thr_tokens_s": thr_tokens_s,
        "thr_mb_s": thr_mb_s,
        "pct_global": pct_global,
        "route_global_rate": route_global_rate,
        "mean_window": mean_window,
        "n_pb_nom": n_pb_nom,
        "a_pb": a_pb,
        "fl_nom_s": fl_nom_s,
        "fl_adj_s": fl_adj_s,
        "c_pb_code": c_pb_code,
    }


def eval_split_metrics(
    *,
    model: torch.nn.Module,
    mc: Any,
    d: Any,
    split: str,
    data_seed: int,
    eval_iters: int,
    mB: int,
    device: str,
    autocast: Any,
    policy: RoutingPolicy,
    flops_parts: dict[str, float],
    bytes_per_tok: float,
    default_window: int,
    no_tqdm: bool,
) -> dict[str, float | None]:
    it = d.iter(split, context_size=mc.context_size, batch_size=mB, seed=data_seed, device=device)
    _sync_cuda_if_needed(device)
    t_wall0 = time.perf_counter()
    losses = estimate_loss_sota(
        it,
        eval_iters,
        model,
        policy,
        default_window=default_window,
        bytes_per_token=d.bytes_per_token,
        autocast=autocast,
        desc=None if no_tqdm else split,
    )
    _sync_cuda_if_needed(device)
    wall_s = time.perf_counter() - t_wall0
    return losses_to_summary(losses, mc, wall_s, eval_iters, mB, bytes_per_tok, flops_parts)


def estimate_loss_sota(
    dataset_iter,
    eval_iters: int,
    model: torch.nn.Module,
    policy: RoutingPolicy,
    *,
    default_window: int,
    bytes_per_token: float | None = None,
    autocast=contextlib.nullcontext(),
    desc: str | None = None,
) -> dict[str, Any]:
    model.eval()
    all_losses = collections.defaultdict(util.MeanError)
    it = range(eval_iters)
    if desc is not None:
        it = tqdm(it, total=eval_iters, desc=desc, leave=True)
    with torch.inference_mode():
        for _ in it:
            tokens, targets = next(dataset_iter)
            route = policy(tokens)
            logits_parts = []
            target_parts = []
            patch_global_context = []
            active_global_context = []

            for b in range(tokens.shape[0]):
                use_global = bool(route["use_global_blocks"][b].item())
                window = int(route["local_attention_window"][b].item())
                set_spacebyte_runtime_params(model, use_global_blocks=use_global, local_window=window)
                target_b = targets[b : b + 1].clone()
                with autocast:
                    logits_b, losses_b = model(tokens[b : b + 1], target_b)
                logits_parts.append(logits_b)
                target_parts.append(target_b)
                gc = losses_b.get("global context", torch.tensor(0.0, device=tokens.device))
                gc_t = torch.as_tensor(gc, device=tokens.device, dtype=torch.float32)
                patch_global_context.append(gc_t)
                active_global_context.append(gc_t if use_global else gc_t * 0.0)

            set_spacebyte_runtime_params(model, use_global_blocks=True, local_window=default_window)
            logits = torch.cat(logits_parts, dim=0)
            routed_targets = torch.cat(target_parts, dim=0)
            losses = {
                "cross entropy": util.cross_entropy(logits, routed_targets, ignore_index=-1),
                "loss": util.cross_entropy(logits, routed_targets, ignore_index=-1),
                "token_XE": util.cross_entropy(logits, routed_targets, reduction="batch", ignore_index=-1),
                "patch global context": torch.stack(patch_global_context).mean(),
                "active global context": torch.stack(active_global_context).mean(),
                "ignored fraction": (routed_targets == -1).float().mean(),
                "route global rate": route["use_global_blocks"].float().mean(),
                "route local attention window": route["local_attention_window"].float().mean(),
                "route probability global": route["prob_global"].float().mean(),
                "route window ratio": route["window_ratio"].float().mean(),
            }
            # Preserve validate.py's summary lookup name while making it adaptive.
            losses["global context"] = losses["active global context"]

            losses = util.tensor_items(losses, dtype=torch.float64)
            for name, loss in losses.items():
                all_losses[name].add(loss)

    for name, losses in list(all_losses.items()):
        all_losses[name] = losses.mean()
        all_losses[name + " stat"] = losses.error()

    if bytes_per_token is not None:
        bpb_mult = 1 / (bytes_per_token * math.log(2))
        all_losses["bits per byte"] = bpb_mult * all_losses["cross entropy"]
        all_losses["bits per byte stat"] = bpb_mult * all_losses["cross entropy stat"]
    return util.tensor_items(all_losses)


def set_spacebyte_runtime_params(
    model: torch.nn.Module,
    *,
    use_global_blocks: bool,
    local_window: int,
) -> None:
    w = int(local_window)
    if w <= 0:
        raise ValueError(f"local_window must be positive, got {local_window}")
    model.config.use_global_blocks = bool(use_global_blocks)
    model.config.local_attention_window = w
    for block in model.initial_blocks:
        block.attention.attention_window = w
    for block in model.final_blocks:
        block.attention.attention_window = w


def spacebyte_utf8_last_byte_use_global(tokens: torch.Tensor, bos: int) -> torch.Tensor:
    raw = spacebyte_utf8_patch_mask(tokens)
    out = raw[:, -1].clone()
    if tokens.shape[1] > 1:
        out &= raw[:, -2].bitwise_not()
    out |= tokens[:, -1] == int(bos)
    return out


def spacebyte_utf8_patch_mask(tokens: torch.Tensor) -> torch.Tensor:
    return (
        (tokens < ord("0"))
        | ((ord("9") < tokens) & (tokens < ord("A")))
        | ((ord("Z") < tokens) & (tokens < ord("a")))
        | ((ord("z") < tokens) & (tokens < 0b1000_0000))
        | (0b1100_0000 <= tokens)
    )


def adaptive_flops_parts(mc: Any, mean_window: float) -> dict[str, float]:
    c = mc.copy(local_attention_window=max(1, int(round(float(mean_window)))))
    return _spacebyte_theoretical_flops_parts(c, True)


def _slug(s: str) -> str:
    return re.sub(r"[^0-9a-zA-Z_\-]+", "_", s).strip("_") or "run"


def run_best_effort(args: argparse.Namespace) -> None:
    if args.router_checkpoint is None:
        raise SystemExit("error: --router-checkpoint is required for --best-effort")
    if args.eval_iters < 3:
        print(
            "warning: eval_iters < 3 breaks variance stats in estimate_loss; using 3.",
            file=sys.stderr,
        )
        args.eval_iters = 3

    os.chdir(_ROOT)
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    import data as data_mod
    import megabyte  # noqa: F401
    import transformer  # noqa: F401
    from megabyte import MegaByte  # noqa: F401
    from spacebyte import SpaceByte, SpaceByteConfig  # noqa: F401
    from transformer import Transformer  # noqa: F401

    ckpt_path = resolve_checkpoint_path(args.checkpoint, args.checkpoint_file)
    if not os.path.isfile(ckpt_path):
        raise SystemExit(f"error: checkpoint file not found: {ckpt_path}")

    checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
    Model = eval(checkpoint["model"])
    raw_cfg = dict(checkpoint["model_config"])
    for key in list(raw_cfg):
        if not hasattr(Model.Config, key):
            del raw_cfg[key]
    cfg = prepare_model_config(Model, raw_cfg, {})
    model, mc = build_model(Model, cfg, checkpoint["state_dict"], device)
    if not isinstance(mc, SpaceByteConfig):
        raise SystemExit("error: validate_sota.py currently supports SpaceByte checkpoints only")

    tc = checkpoint["train_config"]
    d = data_mod.dataset(tc["dataset"], mc.tokenizer)
    model.dataset_tokenizer = d.tokenizer

    bytes_per_tok = float(d.bytes_per_token)
    mB = int(tc["micro_batch_size"])
    autocast = util.autocast_context(tc["dtype"])
    data_seed = int(tc["data_seed"])
    splits = [s.strip() for s in args.splits.split(",") if s.strip()]
    flops_parts = _spacebyte_theoretical_flops_parts(mc, True)
    default_window = int(mc.local_attention_window)

    bundle = load_router_bundle(args.router_checkpoint, main_device=device, router_device=args.router_device)

    thresholds = [round(i / 10.0, 1) for i in range(11)]
    out_dir = Path(args.best_effort_output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    baseline_ns = SimpleNamespace(
        baseline=False,
        no_global_routing=True,
        no_window_routing=True,
        use_global_blocks=True,
        classification_threshold=0.5,
        router_checkpoint=args.router_checkpoint,
        router_device=args.router_device,
    )
    baseline_policy = build_policy(
        baseline_ns, d.tokenizer, default_window, device, router_bundle=None
    )

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    csv_path = out_dir / "best_effort_results.csv"
    rows: list[dict[str, Any]] = []

    for split in splits:
        if split not in d.splits():
            print(f"warning: split '{split}' not in dataset; available: {list(d.splits())}", file=sys.stderr)
            continue

        base_sm = eval_split_metrics(
            model=model,
            mc=mc,
            d=d,
            split=split,
            data_seed=data_seed,
            eval_iters=args.eval_iters,
            mB=mB,
            device=device,
            autocast=autocast,
            policy=baseline_policy,
            flops_parts=flops_parts,
            bytes_per_tok=bytes_per_tok,
            default_window=default_window,
            no_tqdm=args.no_tqdm,
        )
        base_bpb = base_sm["bpb"]
        base_ce = base_sm["ce"]
        base_real = base_sm["c_pb_code"]

        for ng, nw, mode_key in ROUTING_MODES_BEST_EFFORT:
            method_bpbs: list[float] = []
            method_ces: list[float] = []
            method_reals: list[float] = []

            if ng and nw:
                m_ce = float(base_sm["ce"])
                m_bpb = float(base_sm["bpb"]) if base_sm["bpb"] is not None else float("nan")
                m_real = float(base_sm["c_pb_code"])
                for _t in thresholds:
                    rows.append(
                        {
                            "split": split,
                            "mode": mode_key,
                            "no_global_routing": ng,
                            "no_window_routing": nw,
                            "threshold": _t,
                            "baseline_ce": base_ce,
                            "baseline_bpb": base_bpb,
                            "baseline_real_flops_b": base_real,
                            "method_ce": m_ce,
                            "method_bpb": m_bpb,
                            "method_real_flops_b": m_real,
                        }
                    )
                    method_ces.append(m_ce)
                    method_bpbs.append(m_bpb)
                    method_reals.append(m_real)
            else:
                for t in thresholds:
                    pol_ns = SimpleNamespace(
                        baseline=False,
                        no_global_routing=ng,
                        no_window_routing=nw,
                        use_global_blocks=None,
                        classification_threshold=t,
                        router_checkpoint=args.router_checkpoint,
                        router_device=args.router_device,
                    )
                    policy = build_policy(
                        pol_ns, d.tokenizer, default_window, device, router_bundle=bundle
                    )
                    m_sm = eval_split_metrics(
                        model=model,
                        mc=mc,
                        d=d,
                        split=split,
                        data_seed=data_seed,
                        eval_iters=args.eval_iters,
                        mB=mB,
                        device=device,
                        autocast=autocast,
                        policy=policy,
                        flops_parts=flops_parts,
                        bytes_per_tok=bytes_per_tok,
                        default_window=default_window,
                        no_tqdm=args.no_tqdm,
                    )
                    m_ce = float(m_sm["ce"])
                    m_bpb = float(m_sm["bpb"]) if m_sm["bpb"] is not None else float("nan")
                    m_real = float(m_sm["c_pb_code"])
                    rows.append(
                        {
                            "split": split,
                            "mode": mode_key,
                            "no_global_routing": ng,
                            "no_window_routing": nw,
                            "threshold": t,
                            "baseline_ce": base_ce,
                            "baseline_bpb": base_bpb,
                            "baseline_real_flops_b": base_real,
                            "method_ce": m_ce,
                            "method_bpb": m_bpb,
                            "method_real_flops_b": m_real,
                        }
                    )
                    method_ces.append(m_ce)
                    method_bpbs.append(m_bpb)
                    method_reals.append(m_real)

            x = np.arange(len(thresholds))
            w = 0.36
            bb = [float(base_bpb) if base_bpb is not None else float("nan") for _ in thresholds]
            br = [float(base_real) for _ in thresholds]

            fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(12.0, 4.2))
            ax0.bar(x - w / 2, bb, width=w, label="baseline")
            ax0.bar(x + w / 2, method_bpbs, width=w, label="method")
            ax0.set_xticks(x)
            ax0.set_xticklabels([str(t) for t in thresholds])
            ax0.set_xlabel("classification threshold")
            ax0.set_ylabel("bits per byte (BPB)")
            ax0.set_title(f"{split} — {mode_key}: CE (BPB)")
            ax0.legend(loc="best")
            ax0.grid(True, axis="y", alpha=0.3)

            ax1.bar(x - w / 2, br, width=w, label="baseline")
            ax1.bar(x + w / 2, method_reals, width=w, label="method")
            ax1.set_xticks(x)
            ax1.set_xticklabels([str(t) for t in thresholds])
            ax1.set_xlabel("classification threshold")
            ax1.set_ylabel("real FLOPs / byte")
            ax1.set_title(f"{split} — {mode_key}: adaptive FLOPs/B")
            ax1.legend(loc="best")
            ax1.grid(True, axis="y", alpha=0.3)

            fig.tight_layout()
            fig_path = out_dir / f"best_effort_{_slug(split)}_{_slug(mode_key)}.png"
            fig.savefig(fig_path, dpi=150)
            plt.close(fig)

    fieldnames = [
        "split",
        "mode",
        "no_global_routing",
        "no_window_routing",
        "threshold",
        "baseline_ce",
        "baseline_bpb",
        "baseline_real_flops_b",
        "method_ce",
        "method_bpb",
        "method_real_flops_b",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        wcsv = csv.DictWriter(f, fieldnames=fieldnames)
        wcsv.writeheader()
        for r in rows:
            wcsv.writerow({k: r[k] for k in fieldnames})

    md_path = out_dir / "best_effort_table.md"
    with md_path.open("w", encoding="utf-8") as mf:
        mf.write("# validate_sota — best-effort grid\n\n")
        mf.write(f"- checkpoint: `{ckpt_path}`\n")
        mf.write(f"- router: `{args.router_checkpoint}`\n")
        mf.write(f"- eval_iters: {args.eval_iters}\n\n")
        mf.write("| " + " | ".join(fieldnames) + " |\n")
        mf.write("| " + " | ".join(["---"] * len(fieldnames)) + " |\n")

        def fmt_bpb_cell(v: Any) -> str:
            if v is None:
                return "n/a"
            fv = float(v)
            if math.isnan(fv):
                return "n/a"
            return f"{fv:.6f}"

        for r in rows:
            cells = [
                str(r["split"]),
                str(r["mode"]),
                str(r["no_global_routing"]).lower(),
                str(r["no_window_routing"]).lower(),
                str(r["threshold"]),
                f"{r['baseline_ce']:.6f}",
                fmt_bpb_cell(r["baseline_bpb"]),
                f"{r['baseline_real_flops_b']:.6f}",
                f"{r['method_ce']:.6f}",
                fmt_bpb_cell(r["method_bpb"]),
                f"{r['method_real_flops_b']:.6f}",
            ]
            mf.write("| " + " | ".join(cells) + " |\n")

    print(f"best-effort: wrote {csv_path} ({len(rows)} rows), {md_path}, plots under {out_dir}/")


if __name__ == "__main__":
    main()
