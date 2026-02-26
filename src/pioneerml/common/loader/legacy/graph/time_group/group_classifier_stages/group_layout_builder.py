from __future__ import annotations

import numpy as np


class GroupLayoutBuilder:
    """Build graph/group layout tensors from hit-level arrays."""

    def build(self, *, loader, n_rows: int, chunk_in: dict[str, np.ndarray], include_targets: bool) -> dict:
        target_count_candidates: list[np.ndarray] = []
        if include_targets:
            target_count_candidates = [
                loader._counts_from_offsets(chunk_in["group_pion_in_offsets"]),
                loader._counts_from_offsets(chunk_in["group_muon_in_offsets"]),
                loader._counts_from_offsets(chunk_in["group_mip_in_offsets"]),
            ]

        layout = loader._compute_group_layout(
            n_rows=n_rows,
            hits_time_group_offsets=chunk_in["hits_time_group_offsets"],
            hits_time_group_values=chunk_in["hits_time_group_values"],
            row_group_count_candidates=target_count_candidates,
        )

        row_ids_graph, local_gid, graph_event_ids, graph_group_ids = loader._build_graph_index_mapping(
            total_graphs=int(layout["total_graphs"]),
            n_rows=n_rows,
            row_group_counts=layout["row_group_counts"],
            row_group_base=layout["row_group_base"],
            event_ids=chunk_in["event_ids"],
        )

        return {
            "layout": layout,
            "row_ids_graph": row_ids_graph,
            "local_gid": local_gid,
            "graph_event_ids": graph_event_ids,
            "graph_group_ids": graph_group_ids,
        }
