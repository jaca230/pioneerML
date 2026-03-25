from __future__ import annotations

from collections.abc import MutableMapping
from typing import Any

import numpy as np
import torch

from pioneerml.common.data_loader.loaders.array_store import NDArrayStore
from pioneerml.common.data_loader.loaders.stage.stages import EdgeTargetStage


class EventEdgeTargetStage(EdgeTargetStage):
    """Build per-edge binary targets from contributor-id overlap."""

    name = "build_edge_targets"
    requires = ("layout", "edge_index_out")
    provides = ("y_edge",)

    def __init__(
        self,
        *,
        input_state_key: str = "features_in",
        mc_event_field: str = "hits_contrib_mc_event_id",
        cache_templates: bool | None = None,
        cache_max_entries: int | None = None,
    ) -> None:
        self.input_state_key = str(input_state_key)
        self.mc_event_field = str(mc_event_field)
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

    @staticmethod
    def _build_edge_labels_from_mc_event_ids(
        *,
        node_indices: np.ndarray,
        src_local: np.ndarray,
        dst_local: np.ndarray,
        mc_event_id_offsets: np.ndarray,
        mc_event_id_values: np.ndarray,
    ) -> np.ndarray:
        k = int(node_indices.shape[0])
        if k <= 1:
            return np.zeros((src_local.shape[0],), dtype=np.float32)

        mc_to_nodes: dict[int, list[int]] = {}
        for local_i, node_idx in enumerate(node_indices):
            s0 = int(mc_event_id_offsets[node_idx])
            s1 = int(mc_event_id_offsets[node_idx + 1])
            if s1 <= s0:
                continue
            ids = np.unique(mc_event_id_values[s0:s1])
            for mc_id in ids:
                key = int(mc_id)
                if key not in mc_to_nodes:
                    mc_to_nodes[key] = [local_i]
                else:
                    mc_to_nodes[key].append(local_i)

        if not mc_to_nodes:
            return np.zeros((src_local.shape[0],), dtype=np.float32)

        adjacency = np.zeros((k, k), dtype=np.bool_)
        for nodes in mc_to_nodes.values():
            if len(nodes) <= 1:
                continue
            idx = np.asarray(nodes, dtype=np.int64)
            adjacency[np.ix_(idx, idx)] = True

        np.fill_diagonal(adjacency, False)
        return adjacency[src_local, dst_local].astype(np.float32, copy=False)

    def run_loader(self, *, state: MutableMapping[str, Any], owner) -> None:
        if not self.include_targets(owner=owner, state=state):
            state["y_edge"] = None
            return

        layout = state["layout"]
        chunk_in = state.get(self.input_state_key)
        if not isinstance(chunk_in, NDArrayStore):
            raise RuntimeError(
                f"Stage '{self.name}' missing required input state map: {self.input_state_key}"
            )
        if (
            not chunk_in.has_raw(chunk_in.offsets_key(self.mc_event_field, 0))
            or not chunk_in.has_raw(chunk_in.offsets_key(self.mc_event_field, 1))
            or not chunk_in.has_raw(chunk_in.values_key(self.mc_event_field))
        ):
            raise RuntimeError("Training mode requires hits_contrib_mc_event_id for event-splitter edge targets.")

        hit_counts = np.asarray(layout["hit_counts"], dtype=np.int64)
        outer_offsets = chunk_in.offsets(self.mc_event_field, 0)
        outer_counts = (outer_offsets[1:] - outer_offsets[:-1]).astype(np.int64, copy=False)
        if not np.array_equal(outer_counts, hit_counts):
            raise RuntimeError("hits_contrib_mc_event_id outer list lengths do not match event hit counts.")

        total_edges = int(layout["total_edges"])
        y_out = np.zeros((total_edges, 1), dtype=np.float32)
        if total_edges <= 0:
            state["y_edge"] = torch.from_numpy(y_out)
            return

        node_ptr = np.asarray(layout["node_ptr"], dtype=np.int64)
        edge_ptr = np.asarray(layout["edge_ptr"], dtype=np.int64)
        mc_offsets = chunk_in.offsets(self.mc_event_field, 1)
        mc_values = chunk_in.values(self.mc_event_field)
        cache_templates = self._resolve_effective_cache_templates(stage_value=self.cache_templates, owner=owner)
        cache_max_entries = self._resolve_effective_cache_max_entries(
            stage_value=self.cache_max_entries,
            owner=owner,
        )

        for k in np.unique(hit_counts):
            k = int(k)
            if k <= 1:
                continue
            src, dst = self._complete_digraph_cached(
                k,
                cache_templates=cache_templates,
                cache_max_entries=cache_max_entries,
            )
            ecount = int(src.shape[0])
            events = np.flatnonzero(hit_counts == k)
            if events.size == 0:
                continue
            local_nodes = np.arange(k, dtype=np.int64)
            for event_idx in events:
                base = int(node_ptr[event_idx])
                edge_base = int(edge_ptr[event_idx])
                labels = self._build_edge_labels_from_mc_event_ids(
                    node_indices=(base + local_nodes),
                    src_local=src,
                    dst_local=dst,
                    mc_event_id_offsets=mc_offsets,
                    mc_event_id_values=mc_values,
                )
                y_out[edge_base : edge_base + ecount, 0] = labels

        state["y_edge"] = torch.from_numpy(y_out)
