from __future__ import annotations

from collections.abc import MutableMapping
from typing import Any

import numpy as np

from ...array_store.ndarray_store import NDArrayStore
from .base_stage import BaseStage


class GraphFeatureStage(BaseStage):
    """Base stage for graph-level feature construction."""

    name = "build_graph_features"
    requires = ("layout", "local_gid", "row_ids_graph")

    def __init__(self, *, input_state_key: str = "features_in") -> None:
        self.input_state_key = str(input_state_key)

    def get_input_store(self, *, state: MutableMapping[str, Any]) -> NDArrayStore:
        chunk_in = state.get(self.input_state_key)
        if not isinstance(chunk_in, NDArrayStore):
            raise RuntimeError(f"Stage '{self.name}' missing required input state map: {self.input_state_key}")
        return chunk_in

    @staticmethod
    def fill_graph_column_from_group_values(
        *,
        out: np.ndarray,
        dst_col: int,
        vals: np.ndarray,
        offs: np.ndarray,
        total_graphs: int,
        local_gid: np.ndarray,
        row_ids_graph: np.ndarray,
    ) -> None:
        if total_graphs == 0 or vals.size == 0:
            return
        counts = (offs[1:] - offs[:-1]).astype(np.int64, copy=False)
        valid = local_gid < counts[row_ids_graph]
        if not np.any(valid):
            return
        idx = offs[row_ids_graph[valid]] + local_gid[valid]
        out[valid, dst_col] = vals[idx].astype(np.float32, copy=False)

    @staticmethod
    def graph_weighted_sum_from_nodes(*, global_group_id: np.ndarray, values: np.ndarray, total_graphs: int) -> np.ndarray:
        if total_graphs <= 0 or values.size == 0:
            return np.zeros((int(total_graphs),), dtype=np.float32)
        return np.bincount(
            global_group_id.astype(np.int64, copy=False),
            weights=values.astype(np.float64, copy=False),
            minlength=int(total_graphs),
        ).astype(np.float32, copy=False)
