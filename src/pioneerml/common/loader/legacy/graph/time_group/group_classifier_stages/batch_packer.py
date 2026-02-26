from __future__ import annotations

import torch


class BatchPacker:
    """Pack chunk arrays into the canonical chunk dictionary structure."""

    def pack(
        self,
        *,
        n_rows: int,
        total_graphs: int,
        x_out,
        edge_index_out,
        edge_attr_out,
        tgroup_out,
        graph_event_ids,
        graph_group_ids,
        node_ptr,
        edge_ptr,
        targets_torch: torch.Tensor | None,
    ) -> dict:
        chunk_out = {
            "x": torch.from_numpy(x_out),
            "edge_index": torch.from_numpy(edge_index_out),
            "edge_attr": torch.from_numpy(edge_attr_out),
            "time_group_ids": torch.from_numpy(tgroup_out),
            "graph_event_ids": torch.from_numpy(graph_event_ids),
            "graph_group_ids": torch.from_numpy(graph_group_ids),
            "node_ptr": torch.from_numpy(node_ptr),
            "edge_ptr": torch.from_numpy(edge_ptr),
            "num_rows": int(n_rows),
            "num_graphs": int(total_graphs),
        }
        if targets_torch is not None:
            chunk_out["targets"] = targets_torch
        return chunk_out
