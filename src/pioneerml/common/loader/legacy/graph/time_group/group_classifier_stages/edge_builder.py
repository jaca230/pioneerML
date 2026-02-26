from __future__ import annotations

import numpy as np


class EdgeBuilder:
    """Build edge index/attribute arrays from grouped node tensors."""

    def build(
        self,
        *,
        loader,
        layout: dict,
        coord_sorted: np.ndarray,
        z_sorted: np.ndarray,
        e_sorted: np.ndarray,
        view_sorted: np.ndarray,
    ) -> dict:
        total_edges = int(layout["total_edges"])
        edge_index_out = np.empty((2, total_edges), dtype=np.int64)
        edge_attr_out = np.empty((total_edges, loader.EDGE_FEATURE_DIM), dtype=np.float32)

        loader._populate_graph_edges(
            node_counts=layout["node_counts"],
            node_ptr=layout["node_ptr"],
            edge_ptr=layout["edge_ptr"],
            sorted_coord_values=coord_sorted,
            sorted_z_values=z_sorted,
            sorted_edep_values=e_sorted,
            sorted_strip_type_values=view_sorted,
            edge_index_out=edge_index_out,
            edge_attr_out=edge_attr_out,
        )

        return {
            "edge_index_out": edge_index_out,
            "edge_attr_out": edge_attr_out,
        }
