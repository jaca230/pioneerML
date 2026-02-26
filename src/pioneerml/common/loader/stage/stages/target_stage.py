from __future__ import annotations

from collections.abc import MutableMapping, Sequence
from typing import Any

import numpy as np
import torch

from ...array_store.ndarray_store import NDArrayStore
from .base_stage import BaseStage

TargetSpec = tuple[str, int]


class TargetStage(BaseStage):
    """Base stage that builds graph-level target tensors."""

    name = "build_targets"
    requires = ("layout", "local_gid", "row_ids_graph")
    provides = ("targets_torch",)

    def __init__(
        self,
        *,
        target_specs: Sequence[TargetSpec],
        num_classes: int | None = None,
        source_state_key: str = "chunk_in",
    ) -> None:
        self.target_specs = tuple(target_specs)
        self.num_classes = num_classes
        self.source_state_key = str(source_state_key)

    def include_targets(self, *, loader, state: MutableMapping[str, Any]) -> bool:
        _ = state
        return bool(getattr(loader, "include_targets", False))

    @staticmethod
    def _fill_target_column_from_group_values(
        *,
        y_out: np.ndarray,
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
        y_out[valid, dst_col] = vals[idx].astype(np.float32, copy=False)

    def _resolve_num_classes(self) -> int:
        if self.num_classes is not None:
            return int(self.num_classes)
        if not self.target_specs:
            return 0
        return int(max(dst_col for _, dst_col in self.target_specs) + 1)

    def run(self, *, state: MutableMapping[str, Any], loader) -> None:
        if not self.include_targets(loader=loader, state=state):
            state["targets_torch"] = None
            return

        chunk_in = state.get(self.source_state_key)
        if not isinstance(chunk_in, NDArrayStore):
            raise RuntimeError(
                f"Stage '{self.name}' missing required target source state map: {self.source_state_key}"
            )
        layout = state["layout"]
        local_gid = state["local_gid"]
        row_ids_graph = state["row_ids_graph"]
        total_graphs = int(layout["total_graphs"])
        num_classes = self._resolve_num_classes()

        y_out = np.zeros((total_graphs, num_classes), dtype=np.float32)
        for field_name, dst_col in self.target_specs:
            self._fill_target_column_from_group_values(
                y_out=y_out,
                dst_col=int(dst_col),
                vals=chunk_in.values(field_name),
                offs=chunk_in.offsets(field_name, 0),
                total_graphs=total_graphs,
                local_gid=local_gid,
                row_ids_graph=row_ids_graph,
            )
        state["targets_torch"] = torch.from_numpy(y_out)
