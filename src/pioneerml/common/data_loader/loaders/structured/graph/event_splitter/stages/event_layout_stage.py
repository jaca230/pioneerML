from __future__ import annotations

from collections.abc import MutableMapping
from typing import Any

import numpy as np

from pioneerml.common.data_loader.loaders.array_store import NDArrayStore
from pioneerml.common.data_loader.loaders.stage.stages import BaseStage


class EventLayoutStage(BaseStage):
    """Build event-level graph layout for event-splitter workloads."""

    name = "build_layout"
    requires = ("n_rows",)
    provides = ("layout", "graph_event_id", "group_ptr_out")

    def __init__(
        self,
        *,
        input_state_key: str = "features_in",
        hits_time_group_field: str = "hits_time_group",
        use_group_probs: bool = True,
        use_endpoint_preds: bool = True,
        endpoint_quantile_columns: tuple[str, ...] = (),
        endpoint_base_columns: tuple[str, ...] = (),
    ) -> None:
        self.input_state_key = str(input_state_key)
        self.hits_time_group_field = str(hits_time_group_field)
        self.use_group_probs = bool(use_group_probs)
        self.use_endpoint_preds = bool(use_endpoint_preds)
        self.endpoint_quantile_columns = tuple(str(v) for v in endpoint_quantile_columns)
        self.endpoint_base_columns = tuple(str(v) for v in endpoint_base_columns)

    @staticmethod
    def _counts_from_offsets(offsets: np.ndarray) -> np.ndarray:
        return (offsets[1:] - offsets[:-1]).astype(np.int64, copy=False)

    @classmethod
    def _group_counts_from_hits(
        cls,
        *,
        n_rows: int,
        hit_offsets: np.ndarray,
        hit_time_group_values: np.ndarray,
        hit_counts: np.ndarray,
    ) -> np.ndarray:
        if hit_time_group_values.size == 0:
            return np.zeros((n_rows,), dtype=np.int64)
        starts = hit_offsets[:-1].astype(np.int64, copy=False)
        safe_starts = np.minimum(starts, np.maximum(0, hit_time_group_values.size - 1))
        row_tg_max = np.maximum.reduceat(hit_time_group_values, safe_starts).astype(np.int64, copy=False)
        row_tg_max[hit_counts == 0] = -1
        return row_tg_max + 1

    @staticmethod
    def _validate_aligned_counts(
        *,
        observed: np.ndarray,
        expected: np.ndarray,
        label: str,
    ) -> None:
        if not np.array_equal(observed, expected):
            raise RuntimeError(f"{label} list lengths do not match inferred row group counts.")

    def run_loader(self, *, state: MutableMapping[str, Any], owner) -> None:
        _ = owner
        chunk_in = state.get(self.input_state_key)
        if not isinstance(chunk_in, NDArrayStore):
            raise RuntimeError(
                f"Stage '{self.name}' missing required input state map: {self.input_state_key}"
            )
        n_rows = int(state["n_rows"])
        hit_offsets = chunk_in.offsets(self.hits_time_group_field, 0)
        hit_time_group_values = chunk_in.values(self.hits_time_group_field)

        hit_counts = self._counts_from_offsets(hit_offsets)
        total_nodes = int(hit_counts.sum())
        node_ptr = np.zeros((n_rows + 1,), dtype=np.int64)
        node_ptr[1:] = np.cumsum(hit_counts)

        edge_counts = hit_counts * np.maximum(hit_counts - 1, 0)
        total_edges = int(edge_counts.sum())
        edge_ptr = np.zeros((n_rows + 1,), dtype=np.int64)
        edge_ptr[1:] = np.cumsum(edge_counts)

        row_ids_hit = np.repeat(np.arange(n_rows, dtype=np.int64), hit_counts)
        hits_group_counts = self._group_counts_from_hits(
            n_rows=n_rows,
            hit_offsets=hit_offsets,
            hit_time_group_values=hit_time_group_values,
            hit_counts=hit_counts,
        )
        row_group_counts = hits_group_counts.astype(np.int64, copy=False)

        if self.use_group_probs and chunk_in.has_raw(chunk_in.offsets_key("pred_pion", 0)):
            self._validate_aligned_counts(
                observed=self._counts_from_offsets(chunk_in.offsets("pred_pion", 0)),
                expected=row_group_counts,
                label="pred_pion",
            )
        if self.use_endpoint_preds:
            if self.endpoint_quantile_columns and chunk_in.has_raw(chunk_in.offsets_key(self.endpoint_quantile_columns[0], 0)):
                self._validate_aligned_counts(
                    observed=self._counts_from_offsets(chunk_in.offsets(self.endpoint_quantile_columns[0], 0)),
                    expected=row_group_counts,
                    label=self.endpoint_quantile_columns[0],
                )
            elif self.endpoint_base_columns and chunk_in.has_raw(chunk_in.offsets_key(self.endpoint_base_columns[0], 0)):
                self._validate_aligned_counts(
                    observed=self._counts_from_offsets(chunk_in.offsets(self.endpoint_base_columns[0], 0)),
                    expected=row_group_counts,
                    label=self.endpoint_base_columns[0],
                )

        group_ptr = np.zeros((n_rows + 1,), dtype=np.int64)
        group_ptr[1:] = np.cumsum(row_group_counts)
        total_groups = int(group_ptr[-1])
        row_group_base = group_ptr[:-1]

        if total_nodes > 0 and total_groups > 0:
            global_group_id = row_group_base[row_ids_hit] + hit_time_group_values
        else:
            global_group_id = np.zeros((0,), dtype=np.int64)

        state["layout"] = {
            "hit_counts": hit_counts,
            "total_nodes": total_nodes,
            "total_edges": total_edges,
            "total_graphs": int(n_rows),
            "total_groups": total_groups,
            "row_group_counts": row_group_counts,
            "row_group_base": row_group_base,
            "global_group_id": global_group_id,
            "node_ptr": node_ptr,
            "edge_ptr": edge_ptr,
            "group_ptr": group_ptr,
        }
        state["graph_event_id"] = np.arange(n_rows, dtype=np.int64)
        state["group_ptr_out"] = group_ptr
