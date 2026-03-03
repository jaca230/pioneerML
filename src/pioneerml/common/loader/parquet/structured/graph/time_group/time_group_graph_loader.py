from __future__ import annotations

from ..graph_loader import GraphLoader
from .....stage.stages import GraphLayoutStage


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

    def _slice_chunk_batch(self, chunk: dict, g0: int, g1: int):
        d = super()._slice_chunk_batch(chunk, g0, g1)
        if "graph_event_id" not in d:
            d.graph_event_id = d.node_graph_id.new_empty((0,))
        if "graph_time_group_id" not in d:
            d.graph_time_group_id = d.node_graph_id.new_empty((0,))
        d.num_groups = int(d.num_graphs)
        return d
