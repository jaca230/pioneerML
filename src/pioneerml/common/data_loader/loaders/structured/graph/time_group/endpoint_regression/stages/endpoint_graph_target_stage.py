from __future__ import annotations

from collections.abc import MutableMapping
from typing import Any

import numpy as np
import torch

from pioneerml.common.data_loader.loaders.array_store import NDArrayStore
from pioneerml.common.data_loader.loaders.stage.stages import GraphTargetStage


class EndpointGraphTargetStage(GraphTargetStage):
    """Build base graph-level endpoint targets `[start_xyz, end_xyz]` (6 dims)."""

    name = "build_targets"
    requires = ("layout", "local_gid", "row_ids_graph")
    provides = ("y_graph",)

    def __init__(
        self,
        *,
        source_state_key: str = "features_in",
    ) -> None:
        super().__init__(
            target_specs=(
                ("group_start_x", 0),
                ("group_start_y", 1),
                ("group_start_z", 2),
                ("group_end_x", 3),
                ("group_end_y", 4),
                ("group_end_z", 5),
            ),
            num_classes=6,
            source_state_key=source_state_key,
        )

    def run_loader(self, *, state: MutableMapping[str, Any], owner) -> None:
        if not self.include_targets(owner=owner, state=state):
            state["y_graph"] = None
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

        y_out = np.zeros((total_graphs, 6), dtype=np.float32)
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
        state["y_graph"] = torch.from_numpy(y_out)
