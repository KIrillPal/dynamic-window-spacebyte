from __future__ import annotations

import sqlite3
import sys
from dataclasses import dataclass, field
from pathlib import Path
from collections.abc import Sequence
from typing import Any

import torch
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizerBase

SPACEBYTE_ROOT = Path(__file__).resolve().parents[1]
if str(SPACEBYTE_ROOT) not in sys.path:
    sys.path.insert(0, str(SPACEBYTE_ROOT))

import data as spacebyte_data  # noqa: E402


@dataclass(frozen=True)
class GTRow:
    sample_idx: int
    position: int
    use_global_blocks: int
    local_window_fraction: float | None = None
    generation_seed: int = 0
    db_path: str = field(default="", compare=False)

    @property
    def has_regression(self) -> bool:
        return self.local_window_fraction is not None


@dataclass(frozen=True)
class DbSpec:
    path: str
    generation_seed: int
    #: If True, ignore ``local_window_fraction`` in SQLite (placeholders like window size).
    classification_only: bool = False


@dataclass(frozen=True)
class SourceSpec:
    dataset: str
    tokenizer: str | None
    context_size: int


class CombinedSpaceByteRouterDataset(Dataset):
    """Router dataset backed by one or more GT SQLite files.

    The GT generation script stores labels plus ``sample_idx``/``position`` but not
    the text prefix. This dataset replays SpaceByte's deterministic memmap iterator
    with the same seed and decodes the prefix before feeding it to MiniLM.

    Duplicate rows are identified by ``(sample_idx, position)`` within the selected
    split. Exact duplicates are dropped; conflicting duplicates raise an error.
    """

    def __init__(
        self,
        *,
        db_specs: Sequence[DbSpec] | None = None,
        db_paths: Sequence[str] | None = None,
        split: str,
        hf_tokenizer: PreTrainedTokenizerBase,
        source: SourceSpec | None = None,
        generation_seed: int = 0,
        max_length: int = 384,
        limit: int | None = None,
    ) -> None:
        self.db_specs = list(db_specs) if db_specs is not None else [
            DbSpec(path=path, generation_seed=generation_seed, classification_only=False)
            for path in (db_paths or [])
        ]
        self.db_paths = [spec.path for spec in self.db_specs]
        self.split = split
        self.hf_tokenizer = hf_tokenizer
        self.max_length = max_length
        self.rows, self.duplicates_dropped = _load_deduped_rows(self.db_specs, split, limit)
        if not self.rows:
            raise ValueError(f"No rows found for split={split!r} in {self.db_paths}")

        self.source = source or infer_source_spec(self.db_paths, split)
        self.encodings = self._build_encodings()

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        enc = self.encodings[idx]
        row = self.rows[idx]
        out = {
            "input_ids": enc["input_ids"].clone(),
            "attention_mask": enc["attention_mask"].clone(),
            "use_global_blocks": torch.tensor(float(row.use_global_blocks), dtype=torch.float32),
            "local_window_fraction": torch.tensor(
                0.0 if row.local_window_fraction is None else row.local_window_fraction,
                dtype=torch.float32,
            ),
            "has_regression": torch.tensor(float(row.has_regression), dtype=torch.float32),
        }
        if "token_type_ids" in enc:
            out["token_type_ids"] = enc["token_type_ids"].clone()
        return out

    def _build_encodings(self) -> list[dict[str, torch.Tensor]]:
        source_dataset = spacebyte_data.dataset(self.source.dataset, self.source.tokenizer)
        encodings: list[dict[str, torch.Tensor] | None] = [None] * len(self.rows)
        row_indices_by_seed: dict[int, list[int]] = {}
        for idx, row in enumerate(self.rows):
            row_indices_by_seed.setdefault(row.generation_seed, []).append(idx)

        for generation_seed, row_indices in row_indices_by_seed.items():
            data_iter = source_dataset.iter(
                self.split,
                context_size=self.source.context_size,
                batch_size=1,
                seed=generation_seed,
                device="cpu",
            )

            expected_next = 0
            cached_sample_idx = -1
            cached_tokens: torch.Tensor | None = None
            for idx in sorted(row_indices, key=lambda i: (self.rows[i].sample_idx, self.rows[i].position)):
                row = self.rows[idx]
                if row.sample_idx != cached_sample_idx:
                    for _ in range(expected_next, row.sample_idx + 1):
                        cached_tokens, _targets = next(data_iter)
                    expected_next = row.sample_idx + 1
                    cached_sample_idx = row.sample_idx
                assert cached_tokens is not None

                if row.position >= cached_tokens.shape[1]:
                    raise ValueError(
                        f"Row position={row.position} exceeds context size {cached_tokens.shape[1]}"
                    )

                prefix_tokens = cached_tokens[0, : row.position + 1]
                text = source_dataset.tokenizer.decode(_drop_bos(prefix_tokens, source_dataset.tokenizer.BOS))
                enc = self.hf_tokenizer(
                    text,
                    max_length=self.max_length,
                    padding="max_length",
                    truncation=True,
                    return_tensors="pt",
                )
                encodings[idx] = {k: v.squeeze(0).cpu() for k, v in enc.items()}

        return [enc for enc in encodings if enc is not None]


class SpaceByteRouterDataset(CombinedSpaceByteRouterDataset):
    """Backward-compatible single-SQLite wrapper."""

    def __init__(
        self,
        *,
        db_path: str,
        split: str,
        hf_tokenizer: PreTrainedTokenizerBase,
        source: SourceSpec | None = None,
        generation_seed: int = 0,
        max_length: int = 384,
        limit: int | None = None,
    ) -> None:
        super().__init__(
            db_specs=[DbSpec(path=db_path, generation_seed=generation_seed, classification_only=False)],
            split=split,
            hf_tokenizer=hf_tokenizer,
            source=source,
            max_length=max_length,
            limit=limit,
        )


def _load_deduped_rows(
    db_specs: Sequence[DbSpec],
    split: str,
    limit: int | None,
) -> tuple[list[GTRow], int]:
    by_key: dict[tuple[int, int, int], GTRow] = {}
    duplicates_dropped = 0
    for spec in db_specs:
        db_path = spec.path
        with sqlite3.connect(db_path) as conn:
            _require_spacebyte_gt_table(conn, db_path)
            cols = _sqlite_table_columns(conn, "spacebyte_gt")
            required = {"sample_idx", "position", "use_global_blocks"}
            missing = required - cols
            if missing:
                raise ValueError(f"SQLite file {db_path!r} misses required columns: {sorted(missing)}")

            has_fraction_col = "local_window_fraction" in cols
            fraction_expr = "local_window_fraction" if has_fraction_col else "NULL AS local_window_fraction"
            sql = f"""
                SELECT sample_idx, position, use_global_blocks, {fraction_expr}
                FROM spacebyte_gt
                WHERE split = ?
                ORDER BY sample_idx ASC, id ASC
            """
            for sample_idx, position, use_global_blocks, local_window_fraction in conn.execute(sql, (split,)).fetchall():
                if spec.classification_only:
                    frac: float | None = None
                else:
                    frac = None if local_window_fraction is None else float(local_window_fraction)
                row = GTRow(
                    sample_idx=int(sample_idx),
                    position=int(position),
                    use_global_blocks=int(use_global_blocks),
                    local_window_fraction=frac,
                    generation_seed=int(spec.generation_seed),
                    db_path=db_path,
                )
                key = (row.generation_seed, row.sample_idx, row.position)
                prev = by_key.get(key)
                if prev is None:
                    by_key[key] = row
                elif _rows_compatible(prev, row):
                    by_key[key] = _merge_duplicate_rows(prev, row)
                    duplicates_dropped += 1
                else:
                    raise ValueError(
                        "Conflicting duplicate GT row for "
                        f"split={split!r}, seed={row.generation_seed}, "
                        f"sample_idx={row.sample_idx}, position={row.position}"
                    )

    rows = sorted(by_key.values(), key=lambda row: (row.generation_seed, row.sample_idx, row.position))
    if limit is not None:
        rows = rows[: int(limit)]
    return rows, duplicates_dropped


def _rows_compatible(a: GTRow, b: GTRow) -> bool:
    if (
        a.sample_idx != b.sample_idx
        or a.position != b.position
        or a.use_global_blocks != b.use_global_blocks
        or a.generation_seed != b.generation_seed
    ):
        return False
    if a.local_window_fraction is None or b.local_window_fraction is None:
        return True
    return abs(a.local_window_fraction - b.local_window_fraction) <= 1e-12


def _merge_duplicate_rows(a: GTRow, b: GTRow) -> GTRow:
    if a.local_window_fraction is not None:
        return a
    if b.local_window_fraction is not None:
        return b
    return a


def infer_source_spec(db_paths: str | Sequence[str], split: str) -> SourceSpec:
    paths = [db_paths] if isinstance(db_paths, str) else list(db_paths)
    row = None
    for db_path in paths:
        with sqlite3.connect(db_path) as conn:
            _require_spacebyte_gt_table(conn, db_path)
            if "checkpoint" not in _sqlite_table_columns(conn, "spacebyte_gt"):
                continue
            row = conn.execute(
                """
                SELECT checkpoint
                FROM spacebyte_gt
                WHERE split = ?
                ORDER BY id ASC
                LIMIT 1
                """,
                (split,),
            ).fetchone()
        if row is not None:
            break
    if row is None:
        raise ValueError(
            f"No checkpoint row found for split={split!r} in {paths}. "
            "Pass source_dataset/source_context_size explicitly if all DBs are "
            "classification-only and do not store checkpoint paths."
        )

    checkpoint_path = row[0]
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    train_config: dict[str, Any] = checkpoint["train_config"]
    model_config: dict[str, Any] = checkpoint["model_config"]
    return SourceSpec(
        dataset=str(train_config["dataset"]),
        tokenizer=model_config.get("tokenizer"),
        context_size=int(model_config["context_size"]),
    )


def _drop_bos(tokens: torch.Tensor, bos: int) -> torch.Tensor:
    return tokens[tokens != bos]


def _require_spacebyte_gt_table(conn: sqlite3.Connection, db_path: str) -> None:
    has_table = conn.execute(
        """
        SELECT 1
        FROM sqlite_master
        WHERE type = 'table' AND name = 'spacebyte_gt'
        LIMIT 1
        """
    ).fetchone()
    if has_table is None:
        raise ValueError(
            f"SQLite file {db_path!r} does not contain table 'spacebyte_gt'. "
            "Check --db-path; the current train/val GT file is usually "
            "'gt_trainval_v1.sqlite'."
        )


def _sqlite_table_columns(conn: sqlite3.Connection, table: str) -> set[str]:
    return {row[1] for row in conn.execute(f"PRAGMA table_info({table})").fetchall()}
