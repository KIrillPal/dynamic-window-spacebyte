#!/usr/bin/env python3
from __future__ import annotations

import argparse
import contextlib
import os
import sys
from pathlib import Path

import torch
from tqdm import tqdm
from transformers import AutoTokenizer

import util
from generate_gt_dataset import (
    compute_use_global_and_min_window,
    greedy_next_token,
)
from router.dataset import SpaceByteRouterDataset, spacebyte_data

ROOT = Path(__file__).resolve().parent

DEFAULT_CHECKPOINT = (
    "/home/kondrashov_k/mipt/hw/nlp/spacebyte/spacebyte-200M-v2/"
    "Train--batch_size=64--beta2=0.98--context_size=2048--d_local=384--d_model=1024"
    "--dataset=pg19--device=cuda:0--global_context_size=1024--iters=3e9/tokens"
    "--local_attention_window=384--lr=0.5e-2*B**0.5--micro_batch_size=4"
    "--model=SpaceByte--n_layers=12--n_local_layers=12--out_dir=spacebyte-200M-v2"
    "--patch_method=utf8--rope=True/ckpt_best_loss.pt"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Check that SpaceByteRouterDataset labels match labels recomputed "
            "with generate_gt_dataset.py logic."
        )
    )
    parser.add_argument("--db-path", default="gt_trainval_v1.sqlite")
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
    parser.add_argument("--checkpoint-file", default=None)
    parser.add_argument("--split", default="val")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--margin", type=float, default=30.0)
    parser.add_argument(
        "--window-search",
        choices=("linear", "binary", "none"),
        default="binary",
    )
    parser.add_argument("--num-samples", type=int, default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--model-name", default="sentence-transformers/all-MiniLM-L6-v2")
    parser.add_argument("--max-length", type=int, default=384)
    parser.add_argument("--fraction-tol", type=float, default=1e-6)
    parser.add_argument("--print-matches", action="store_true")
    parser.add_argument("--max-text-preview", type=int, default=120)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    os.chdir(ROOT)

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    tokenizer.truncation_side = "left"

    router_dataset = SpaceByteRouterDataset(
        db_path=args.db_path,
        split=args.split,
        hf_tokenizer=tokenizer,
        generation_seed=args.seed,
        max_length=args.max_length,
        limit=args.num_samples,
    )

    model, default_window, autocast = load_spacebyte_model(
        args.checkpoint,
        checkpoint_file=args.checkpoint_file,
        device=device,
    )

    source_dataset = spacebyte_data.dataset(
        router_dataset.source.dataset,
        router_dataset.source.tokenizer,
    )
    data_iter = source_dataset.iter(
        args.split,
        context_size=router_dataset.source.context_size,
        batch_size=1,
        seed=args.seed,
        device=device,
    )

    total = 0
    flag_mismatches = 0
    fraction_mismatches = 0
    any_mismatches = 0
    max_abs_fraction_diff = 0.0
    expected_next = 0

    for idx, row in enumerate(tqdm(router_dataset.rows, desc="checking GT")):
        item = router_dataset[idx]
        tokens = None
        targets = None
        for _ in range(expected_next, row.sample_idx + 1):
            tokens, targets = next(data_iter)
        expected_next = row.sample_idx + 1
        assert tokens is not None and targets is not None

        real_gt = int(targets[0, row.position].item())
        global_gt = greedy_next_token(
            model,
            tokens,
            row.position,
            use_global_blocks=True,
            local_window=default_window,
            autocast_ctx=autocast,
        )
        pred_local_default = greedy_next_token(
            model,
            tokens,
            row.position,
            use_global_blocks=False,
            local_window=default_window,
            autocast_ctx=autocast,
        )
        recomputed_flag, recomputed_min_window, recomputed_fraction, _search = (
            compute_use_global_and_min_window(
                model,
                tokens,
                row.position,
                default_window,
                global_gt,
                real_gt,
                pred_local_default,
                margin=args.margin,
                autocast_ctx=autocast,
                window_search=args.window_search,
            )
        )

        dataset_flag = int(item["use_global_blocks"].item())
        dataset_fraction = float(item["local_window_fraction"].item())
        fraction_diff = abs(dataset_fraction - recomputed_fraction)
        flag_bad = dataset_flag != recomputed_flag
        fraction_bad = fraction_diff > args.fraction_tol
        max_abs_fraction_diff = max(max_abs_fraction_diff, fraction_diff)

        total += 1
        flag_mismatches += int(flag_bad)
        fraction_mismatches += int(fraction_bad)
        any_mismatches += int(flag_bad or fraction_bad)

        if flag_bad or fraction_bad or args.print_matches:
            text_preview = tokenizer.decode(
                item["input_ids"],
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )[-args.max_text_preview :]
            status = "MISMATCH" if flag_bad or fraction_bad else "OK"
            print(
                f"{status} idx={idx} sample_idx={row.sample_idx} pos={row.position} "
                f"flag_dataset={dataset_flag} flag_recomputed={recomputed_flag} "
                f"frac_dataset={dataset_fraction:.12g} "
                f"frac_recomputed={recomputed_fraction:.12g} "
                f"abs_frac_diff={fraction_diff:.3g} "
                f"min_window_recomputed={recomputed_min_window} "
                f"real_gt={real_gt} global_gt={global_gt} "
                f"pred_local_default={pred_local_default} "
                f"text_tail={text_preview!r}",
                flush=True,
            )

    print(
        "summary: "
        f"total={total} "
        f"any_mismatches={any_mismatches} "
        f"flag_mismatches={flag_mismatches} "
        f"fraction_mismatches={fraction_mismatches} "
        f"max_abs_fraction_diff={max_abs_fraction_diff:.12g} "
        f"fraction_tol={args.fraction_tol}",
        flush=True,
    )


def load_spacebyte_model(checkpoint_arg: str, *, checkpoint_file: str | None, device: str):
    import megabyte  # noqa: F401
    import transformer  # noqa: F401
    from megabyte import MegaByte  # noqa: F401
    from spacebyte import SpaceByte, SpaceByteConfig  # noqa: F401
    from transformer import Transformer  # noqa: F401
    from validate import build_model, prepare_model_config, resolve_checkpoint_path

    ckpt_path = resolve_checkpoint_path(checkpoint_arg, checkpoint_file)
    checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
    Model = eval(checkpoint["model"])

    raw_cfg = dict(checkpoint["model_config"])
    for key in list(raw_cfg):
        if not hasattr(Model.Config, key):
            del raw_cfg[key]
    cfg = prepare_model_config(Model, raw_cfg, {})

    cfg_obj = Model.Config(**cfg)
    if not isinstance(cfg_obj, SpaceByteConfig):
        raise TypeError(f"Expected SpaceByte checkpoint, got {type(cfg_obj).__name__}")

    model, mc = build_model(Model, cfg, checkpoint["state_dict"], device)
    default_window = int(mc.local_attention_window)
    autocast = (
        util.autocast_context(checkpoint["train_config"]["dtype"])
        if str(device).startswith("cuda")
        else contextlib.nullcontext()
    )
    return model, default_window, autocast


if __name__ == "__main__":
    main()
