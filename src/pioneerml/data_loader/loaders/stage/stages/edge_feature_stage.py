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

    def __init__(
        self,
        *,
        edge_feature_dim: int = 4,
        edge_populate_graph_block: int | None = None,
        cache_templates: bool | None = None,
        cache_max_entries: int | None = None,
    ) -> None:
        self.edge_feature_dim = int(edge_feature_dim)
        self.edge_populate_graph_block = None if edge_populate_graph_block is None else int(edge_populate_graph_block)
        self.cache_templates = None if cache_templates is None else bool(cache_templates)
        self.cache_max_entries = self._normalize_cache_max_entries(cache_max_entries)
        self._edge_tpl_cache: dict[int, tuple[np.ndarray, np.ndarray]] = {}

    @staticmethod
    def _normalize_cache_max_entries(value: int | None) -> int | None:
        if value is None:
            return None
        out = int(value)
        if out <= 0:
            return 0
        return out

    @staticmethod
    def _resolve_effective_cache_templates(*, stage_value: bool | None, owner) -> bool:
        if stage_value is not None:
            return bool(stage_value)
        return bool(getattr(owner, "edge_template_cache_enabled", False))

    @classmethod
    def _resolve_effective_cache_max_entries(cls, *, stage_value: int | None, owner) -> int | None:
        if stage_value is not None:
            return cls._normalize_cache_max_entries(stage_value)
        return cls._normalize_cache_max_entries(getattr(owner, "edge_template_cache_max_entries", None))

    def _complete_digraph_cached(
        self,
        k: int,
        *,
        cache_templates: bool,
        cache_max_entries: int | None,
    ) -> tuple[np.ndarray, np.ndarray]:
        if not cache_templates or cache_max_entries == 0:
            src = np.repeat(np.arange(k, dtype=np.int64), k)
            dst = np.tile(np.arange(k, dtype=np.int64), k)
            mask = src != dst
            return src[mask], dst[mask]

        tpl = self._edge_tpl_cache.get(k)
        if tpl is not None:
            return tpl
        src = np.repeat(np.arange(k, dtype=np.int64), k)
        dst = np.tile(np.arange(k, dtype=np.int64), k)
        mask = src != dst
        tpl = (src[mask], dst[mask])
        if cache_max_entries is not None and len(self._edge_tpl_cache) >= cache_max_entries:
            oldest = next(iter(self._edge_tpl_cache), None)
            if oldest is not None:
                self._edge_tpl_cache.pop(oldest, None)
        self._edge_tpl_cache[k] = tpl
        return tpl

    def run_loader(self, *, state: MutableMapping[str, Any], owner) -> None:
        layout = state["layout"]
        total_edges = int(layout["total_edges"])
        edge_index_out = np.empty((2, total_edges), dtype=np.int64)
        edge_attr_out = np.empty((total_edges, self.edge_feature_dim), dtype=np.float32)
        if total_edges > 0:
            cache_templates = self._resolve_effective_cache_templates(stage_value=self.cache_templates, owner=owner)
            cache_max_entries = self._resolve_effective_cache_max_entries(
                stage_value=self.cache_max_entries,
                owner=owner,
            )
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
                else getattr(owner, "edge_populate_graph_block", 512)
            )
            for k in np.unique(node_counts):
                k = int(k)
                if k <= 1:
                    continue
                src, dst = self._complete_digraph_cached(
                    k,
                    cache_templates=cache_templates,
                    cache_max_entries=cache_max_entries,
                )
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
