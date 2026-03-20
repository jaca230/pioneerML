from __future__ import annotations

from ..graph_loader import GraphLoader


class TimeGroupGraphLoader(GraphLoader):
    """Graph loader base for time-group tasks."""

    def data_struct_fields(self) -> tuple[str, ...]:
        fields = list(super().data_struct_fields())
        fields.extend(["graph_event_id", "graph_time_group_id"])
        return tuple(fields)

    def empty_data(self):
        data = super().empty_data()
        data.graph_event_id = data.node_graph_id.new_empty((0,))
        data.graph_time_group_id = data.node_graph_id.new_empty((0,))
        data.num_groups = 0
        return data

    def _slice_chunk_batch(self, chunk: dict, g0: int, g1: int):
        d = super()._slice_chunk_batch(chunk, g0, g1)
        if "graph_event_id" not in d:
            d.graph_event_id = d.node_graph_id.new_empty((0,))
        if "graph_time_group_id" not in d:
            d.graph_time_group_id = d.node_graph_id.new_empty((0,))
        d.num_groups = int(d.num_graphs)
        return d
