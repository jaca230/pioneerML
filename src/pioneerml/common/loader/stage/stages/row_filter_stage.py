from __future__ import annotations

from collections.abc import MutableMapping
from typing import Any

import numpy as np
import pyarrow as pa

from ...utils.hashing import SAMPLE_STREAM_DOMAIN_SEED, keyed_uniform01
from .base_stage import BaseStage


class RowFilterStage(BaseStage):
    """Base row-level deterministic split/sample filter."""

    name = "row_filter"
    requires = ("table",)
    provides = ("table",)

    def __init__(
        self,
        *,
        event_id_column: str = "event_id",
        split: str | None = None,
        train_fraction: float = 0.9,
        val_fraction: float = 0.05,
        split_seed: int = 0,
        sample_fraction: float | None = None,
    ) -> None:
        self.event_id_column = str(event_id_column)
        split_norm = None if split is None else str(split).strip().lower()
        if split_norm is not None and split_norm not in {"train", "val", "test"}:
            raise ValueError(f"Unsupported split: {split}. Expected one of: 'train', 'val', 'test'.")
        self.split = split_norm
        self.train_fraction = float(train_fraction)
        self.val_fraction = float(val_fraction)
        self.split_seed = int(split_seed)
        self.sample_fraction = None if sample_fraction is None else float(sample_fraction)
        if self.sample_fraction is not None and not (0.0 < self.sample_fraction <= 1.0):
            raise ValueError(f"sample_fraction must be in (0, 1], got: {self.sample_fraction}")
        if self.split is not None:
            total = self.train_fraction + self.val_fraction
            if total <= 0.0 or total > 1.0:
                # test fraction is implicit remainder; keep sanity check explicit.
                raise ValueError(
                    "train_fraction + val_fraction must be in (0, 1] when split filtering is enabled."
                )

    @classmethod
    def _row_mask(
        cls,
        *,
        event_ids: np.ndarray,
        split: str | None,
        train_fraction: float,
        val_fraction: float,
        split_seed: int,
        sample_fraction: float | None,
    ) -> np.ndarray:
        n_rows = int(event_ids.shape[0])
        if n_rows == 0:
            return np.zeros((0,), dtype=bool)

        event_u64 = event_ids.astype(np.uint64, copy=False)
        u_split = keyed_uniform01(key_values=event_u64, seed=split_seed)
        mask = np.ones((n_rows,), dtype=bool)

        if split is not None:
            split_norm = str(split).strip().lower()
            train_hi = float(train_fraction)
            val_hi = train_hi + float(val_fraction)
            if split_norm == "train":
                mask &= u_split < train_hi
            elif split_norm == "val":
                mask &= (u_split >= train_hi) & (u_split < val_hi)
            elif split_norm == "test":
                mask &= u_split >= val_hi
            else:
                raise ValueError(f"Unsupported split: {split}. Expected one of: 'train', 'val', 'test'.")

        if sample_fraction is not None:
            u_sample = keyed_uniform01(
                key_values=event_u64,
                seed=0,
                domain_seed=int(SAMPLE_STREAM_DOMAIN_SEED),
            )
            mask &= u_sample < float(sample_fraction)

        return mask

    def run(self, *, state: MutableMapping[str, Any], loader) -> None:
        table = state.get("table")
        if table is None or int(table.num_rows) == 0:
            state["table"] = None
            state["chunk_out"] = None
            state["stop_pipeline"] = True
            return

        split = self.split
        sample_fraction = self.sample_fraction
        if split is None and sample_fraction is None:
            return

        if self.event_id_column not in table.column_names:
            raise RuntimeError(f"{self.event_id_column} column is required for split/sample filtering.")

        event_ids = table.column(self.event_id_column).chunk(0).to_numpy(zero_copy_only=False).astype(np.int64, copy=False)
        mask = self._row_mask(
            event_ids=event_ids,
            split=split,
            train_fraction=self.train_fraction,
            val_fraction=self.val_fraction,
            split_seed=self.split_seed,
            sample_fraction=None if sample_fraction is None else float(sample_fraction),
        )
        if mask.size == 0 or not np.any(mask):
            state["table"] = None
            state["chunk_out"] = None
            state["stop_pipeline"] = True
            return

        if not np.all(mask):
            table = table.filter(pa.array(mask))

        if int(table.num_rows) == 0:
            state["table"] = None
            state["chunk_out"] = None
            state["stop_pipeline"] = True
            return

        state["table"] = table
