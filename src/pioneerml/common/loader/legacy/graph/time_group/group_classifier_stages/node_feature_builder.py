from __future__ import annotations

import numpy as np


class NodeFeatureBuilder:
    """Build node feature and node-level metadata arrays."""

    def build(self, *, loader, chunk_in: dict[str, np.ndarray], layout: dict) -> dict:
        total_nodes = int(layout["total_nodes"])
        x_out = np.empty((total_nodes, loader.NODE_FEATURE_DIM), dtype=np.float32)
        tgroup_out = np.empty((total_nodes,), dtype=np.int64)

        coord_sorted, z_sorted, e_sorted, view_sorted = loader._populate_node_tensors(
            total_nodes=total_nodes,
            global_group_id=layout["global_group_id"],
            hit_coord_values=chunk_in["hits_coord_values"],
            hit_z_values=chunk_in["hits_z_values"],
            hit_edep_values=chunk_in["hits_edep_values"],
            hit_strip_type_values=chunk_in["hits_strip_type_values"],
            hit_time_group_values=chunk_in["hits_time_group_values"],
            x_out=x_out,
            time_group_ids_out=tgroup_out,
        )

        return {
            "x_out": x_out,
            "tgroup_out": tgroup_out,
            "coord_sorted": coord_sorted,
            "z_sorted": z_sorted,
            "e_sorted": e_sorted,
            "view_sorted": view_sorted,
        }
