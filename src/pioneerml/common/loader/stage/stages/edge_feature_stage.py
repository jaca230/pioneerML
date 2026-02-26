from __future__ import annotations

from collections.abc import MutableMapping
from typing import Any

import numpy as np

from .base_stage import BaseStage


class EdgeFeatureStage(BaseStage):
    """Base stage that builds edge index and edge feature arrays."""

    name = "build_edges"
    requires = ("layout", "coord_sorted", "z_sorted", "e_sorted", "view_sorted")
    provides = ("edge_index_out", "edge_attr_out")

    def __init__(self, *, edge_feature_dim: int = 4, edge_populate_graph_block: int | None = None) -> None:
        self.edge_feature_dim = int(edge_feature_dim)
        self.edge_populate_graph_block = None if edge_populate_graph_block is None else int(edge_populate_graph_block)
        self._edge_tpl_cache: dict[int, tuple[np.ndarray, np.ndarray]] = {}

    def _complete_digraph_cached(self, k: int) -> tuple[np.ndarray, np.ndarray]:
        tpl = self._edge_tpl_cache.get(k)
        if tpl is not None:
            return tpl
        src = np.repeat(np.arange(k, dtype=np.int64), k)
        dst = np.tile(np.arange(k, dtype=np.int64), k)
        mask = src != dst
        tpl = (src[mask], dst[mask])
        self._edge_tpl_cache[k] = tpl
        return tpl

    def run(self, *, state: MutableMapping[str, Any], loader) -> None:
        layout = state["layout"]
        total_edges = int(layout["total_edges"])
        edge_index_out = np.empty((2, total_edges), dtype=np.int64)
        edge_attr_out = np.empty((total_edges, self.edge_feature_dim), dtype=np.float32)
        if total_edges > 0:
            node_counts = layout["node_counts"]
            node_ptr = layout["node_ptr"]
            edge_ptr = layout["edge_ptr"]
            coord_sorted = state["coord_sorted"]
            z_sorted = state["z_sorted"]
            e_sorted = state["e_sorted"]
            view_sorted = state["view_sorted"]
            graph_block = int(
                self.edge_populate_graph_block
                if self.edge_populate_graph_block is not None
                else getattr(loader, "edge_populate_graph_block", 512)
            )
            for k in np.unique(node_counts):
                k = int(k)
                if k <= 1:
                    continue
                src, dst = self._complete_digraph_cached(k)
                ecount = int(src.shape[0])
                graphs = np.flatnonzero(node_counts == k)
                if graphs.size == 0:
                    continue
                rel_edge = np.arange(ecount, dtype=np.int64)
                for i in range(0, graphs.size, graph_block):
                    g = graphs[i : i + graph_block]
                    node_base = node_ptr[g]
                    edge_base = edge_ptr[g]
                    pos = (edge_base[:, None] + rel_edge[None, :]).reshape(-1)
                    src_idx = (node_base[:, None] + src[None, :]).reshape(-1)
                    dst_idx = (node_base[:, None] + dst[None, :]).reshape(-1)
                    edge_index_out[0, pos] = src_idx
                    edge_index_out[1, pos] = dst_idx
                    edge_attr_out[pos, 0] = coord_sorted[dst_idx] - coord_sorted[src_idx]
                    edge_attr_out[pos, 1] = z_sorted[dst_idx] - z_sorted[src_idx]
                    edge_attr_out[pos, 2] = e_sorted[dst_idx] - e_sorted[src_idx]
                    edge_attr_out[pos, 3] = (view_sorted[src_idx] == view_sorted[dst_idx]).astype(np.float32)

        state["edge_index_out"] = edge_index_out
        state["edge_attr_out"] = edge_attr_out
