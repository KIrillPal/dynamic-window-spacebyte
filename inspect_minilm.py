#!/usr/bin/env python3
"""
Load sentence-transformers/all-MiniLM-L6-v2 and print module structure.

Requires: pip install transformers

For fine-tuning heads: use contextualized vectors — i.e. the last encoder layer
output (HF: forward() -> last_hidden_state), not the initial Embedding layer.
Mean / max pooling over sequence + mask is typical (Sentence-Transformers does mean pooling).
"""

from __future__ import annotations

import argparse


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        default="sentence-transformers/all-MiniLM-L6-v2",
        help="HF model id",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Print every submodule (long listing)",
    )
    args = parser.parse_args()

    from transformers import AutoConfig, AutoModel

    name = args.model
    config = AutoConfig.from_pretrained(name)
    model = AutoModel.from_pretrained(name)

    print(f"Model: {name}")
    print(f"Architecture: {config.model_type}")
    print(f"Hidden size: {config.hidden_size}")
    print(f"Num encoder layers: {config.num_hidden_layers}")
    print()

    print("=== Top-level children ===")
    for i, (n, m) in enumerate(model.named_children()):
        print(f"  [{i}] {n}: {m.__class__.__name__}")

    print()
    print("=== Encoder stack (one line per layer) ===")
    enc = model.encoder.layer
    for i in range(len(enc)):
        print(f"  encoder.layer.{i}: {enc[i].__class__.__name__}")

    if args.verbose:
        print()
        print("=== All named modules (flat) ===")
        for name_mod, mod in model.named_modules():
            if name_mod == "":
                continue
            print(f"  {name_mod}: {mod.__class__.__name__}")
    else:
        print()
        print("(Use --verbose for full flat module list.)")

    print()
    print(
        "=== What to use for a classification / regression head ===\n"
        "  • embeddings — только token + position (+ optional token_type); БЕЗ контекста\n"
        "    между позициями. Для семантики префикса обычно НЕ берут только это.\n"
        f"  • encoder.layer.0 … encoder.layer.{config.num_hidden_layers - 1} — блоки\n"
        "    Transformer; последний даёт полностью контекстуализированные скрытые состояния.\n"
        "  • В коде Hugging Face: outputs = model(**batch); vec = outputs.last_hidden_state\n"
        "    Shape [batch, seq_len, hidden_size] — это выход ПОСЛЕДНЕГО слоя encoder.\n"
        "  • Дальше: Pooling (mean по маске, CLS и т.д.) -> Linear головы.\n"
        "  • У SentenceTransformer это обёртка: Transformer -> Pooling(mean).\n"
    )


if __name__ == "__main__":
    main()
