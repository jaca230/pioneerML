from __future__ import annotations

import torch
from torch_geometric.data import Data

from ..graph_loader import GraphLoader
from .....stage.stages import GraphLayoutStage


class TimeGroupGraphLoader(GraphLoader):
    """Graph loader base for time-group tasks."""

    def _include_time_group_ids_in_empty_data(self) -> bool:
        return True

    def empty_data(self) -> tuple[Data, torch.Tensor]:
        data, targets = super().empty_data()
        data.num_groups = 0
        return data, targets

    @staticmethod
    def make_time_group_layout_stage(
        *,
        row_group_count_fields: tuple[str, ...] = (),
        input_state_key: str = "chunk_in",
        source_state_keys: tuple[str, ...] | None = None,
        hits_time_group_field: str = "hits_time_group",
    ) -> GraphLayoutStage:
        return GraphLayoutStage(
            input_state_key=input_state_key,
            source_state_keys=source_state_keys,
            hits_time_group_field=hits_time_group_field,
            row_group_count_fields=row_group_count_fields,
        )

    def _slice_chunk_batch(self, chunk: dict, g0: int, g1: int) -> Data:
        node_ptr = chunk["node_ptr"]
        edge_ptr = chunk["edge_ptr"]
        n0 = int(node_ptr[g0].item())
        n1 = int(node_ptr[g1].item())
        e0 = int(edge_ptr[g0].item())
        e1 = int(edge_ptr[g1].item())

        d = Data(
            x=chunk["x"][n0:n1],
            edge_index=(chunk["edge_index"][:, e0:e1] - n0),
            edge_attr=chunk["edge_attr"][e0:e1],
            time_group_ids=chunk["time_group_ids"][n0:n1],
        )
        if "targets" in chunk:
            d.y = chunk["targets"][g0:g1]
        local_counts = (node_ptr[g0 + 1 : g1 + 1] - node_ptr[g0:g1]).to(dtype=torch.int64)
        d.batch = torch.repeat_interleave(torch.arange(g1 - g0, dtype=torch.int64), local_counts)
        d.event_ids = chunk["graph_event_ids"][g0:g1]
        d.group_ids = chunk["graph_group_ids"][g0:g1]
        d.num_graphs = int(g1 - g0)
        d.num_groups = int(g1 - g0)
        return d
