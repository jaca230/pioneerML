from __future__ import annotations

from collections.abc import MutableMapping
from typing import Any

import numpy as np
import pyarrow as pa

from ...config import SplitSampleConfig
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
        split_config: SplitSampleConfig | None = None,
    ) -> None:
        self.event_id_column = str(event_id_column)
        cfg = split_config if split_config is not None else SplitSampleConfig()
        self.split = cfg.split
        self.train_fraction = float(cfg.train_fraction)
        self.val_fraction = float(cfg.val_fraction)
        self.split_seed = int(cfg.split_seed if cfg.split_seed is not None else 0)
        self.sample_fraction = cfg.sample_fraction

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

    def run_loader(self, *, state: MutableMapping[str, Any], owner) -> None:
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
