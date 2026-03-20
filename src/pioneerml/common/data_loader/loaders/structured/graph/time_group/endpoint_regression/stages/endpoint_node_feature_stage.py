from __future__ import annotations

from collections.abc import MutableMapping
from typing import Any

import numpy as np

from pioneerml.common.data_loader.loaders.array_store import NDArrayStore
from pioneerml.common.data_loader.loaders.stage.stages import NodeFeatureStage


class EndpointNodeFeatureStage(NodeFeatureStage):
    """Build endpoint node features, including optional splitter probabilities."""

    name = "build_nodes"
    requires = ("layout",)
    provides = ("x_out", "tgroup_out", "coord_sorted", "z_sorted", "e_sorted", "view_sorted", "group_truth_out")

    def __init__(
        self,
        *,
        input_state_key: str = "features_in",
        coord_field: str = "hits_coord",
        z_field: str = "hits_z",
        edep_field: str = "hits_edep",
        strip_type_field: str = "hits_strip_type",
        time_group_field: str = "hits_time_group",
        particle_mask_field: str = "hits_particle_mask",
        node_feature_dim: int = 7,
    ) -> None:
        super().__init__(
            input_state_key=input_state_key,
            coord_field=coord_field,
            z_field=z_field,
            edep_field=edep_field,
            strip_type_field=strip_type_field,
            time_group_field=time_group_field,
            node_feature_dim=node_feature_dim,
        )
        self.particle_mask_field = str(particle_mask_field)

    @staticmethod
    def _particle_mask_to_multihot(mask_values: np.ndarray) -> np.ndarray:
        out = np.zeros((mask_values.size, 3), dtype=np.float32)
        out[:, 0] = ((mask_values & 1) != 0).astype(np.float32, copy=False)
        out[:, 1] = ((mask_values & 2) != 0).astype(np.float32, copy=False)
        out[:, 2] = ((mask_values & 4) != 0).astype(np.float32, copy=False)
        return out

    @staticmethod
    def _build_splitter_value_index_per_hit(
        *,
        n_rows: int,
        hit_offsets: np.ndarray,
        hit_time_group_values: np.ndarray,
        splitter_offsets: np.ndarray,
        splitter_time_group_offsets: np.ndarray | None,
        splitter_time_group_values: np.ndarray | None,
    ) -> np.ndarray:
        total_hits = int(hit_offsets[-1]) if hit_offsets.size > 0 else 0
        out = np.zeros((total_hits,), dtype=np.int64)
        for row in range(n_rows):
            h0 = int(hit_offsets[row])
            h1 = int(hit_offsets[row + 1])
            n_hits = h1 - h0
            s0 = int(splitter_offsets[row])
            s1 = int(splitter_offsets[row + 1])
            if (s1 - s0) != n_hits:
                raise RuntimeError("Splitter probability list length does not match hits.")
            if n_hits <= 0:
                continue

            local_map = s0 + np.arange(n_hits, dtype=np.int64)
            if splitter_time_group_offsets is not None and splitter_time_group_values is not None:
                tg0 = int(splitter_time_group_offsets[row])
                tg1 = int(splitter_time_group_offsets[row + 1])
                if (tg1 - tg0) != n_hits:
                    out[h0:h1] = local_map
                    continue
                hit_tg = hit_time_group_values[h0:h1]
                split_tg = splitter_time_group_values[tg0:tg1]
                if not np.array_equal(hit_tg, split_tg):
                    for gid in np.unique(hit_tg):
                        hit_pos = np.flatnonzero(hit_tg == gid)
                        split_pos = np.flatnonzero(split_tg == gid)
                        if hit_pos.size != split_pos.size:
                            local_map = s0 + np.arange(n_hits, dtype=np.int64)
                            break
                        local_map[hit_pos] = s0 + split_pos
            out[h0:h1] = local_map
        return out

    def run_loader(self, *, state: MutableMapping[str, Any], owner) -> None:
        layout = state["layout"]
        chunk_in = state.get(self.input_state_key)
        if not isinstance(chunk_in, NDArrayStore):
            raise RuntimeError(f"Stage '{self.name}' missing required input state map: {self.input_state_key}")

        total_nodes = int(layout["total_nodes"])
        total_graphs = int(layout["total_graphs"])
        global_group_id = layout["global_group_id"]

        x_out = np.empty((total_nodes, self.node_feature_dim), dtype=np.float32)
        tgroup_out = np.empty((total_nodes,), dtype=np.int64)
        group_truth_out = np.zeros((total_graphs, 3), dtype=np.float32)

        if total_nodes > 0:
            order = np.argsort(global_group_id, kind="stable")
            coord_sorted = chunk_in.values(self.coord_field)[order]
            z_sorted = chunk_in.values(self.z_field)[order]
            e_sorted = chunk_in.values(self.edep_field)[order]
            view_sorted = chunk_in.values(self.strip_type_field)[order]
            tgroup_sorted = chunk_in.values(self.time_group_field)[order]

            x_out[:, 0] = coord_sorted
            x_out[:, 1] = z_sorted
            x_out[:, 2] = e_sorted
            x_out[:, 3] = view_sorted.astype(np.float32, copy=False)
            tgroup_out[:] = tgroup_sorted

            # Defaults for splitter-probability channels.
            x_out[:, 4:7] = 0.0

            has_splitter_columns = all(
                chunk_in.has_raw(chunk_in.values_key(name)) and chunk_in.has_raw(chunk_in.offsets_key(name, 0))
                for name in ("pred_hit_pion", "pred_hit_muon", "pred_hit_mip")
            )

            if has_splitter_columns:
                hit_offsets = chunk_in.offsets(self.time_group_field, 0)
                hit_time_group_values = chunk_in.values(self.time_group_field)
                splitter_offsets = chunk_in.offsets("pred_hit_pion", 0)
                has_splitter_tg = chunk_in.has_raw(chunk_in.values_key("time_group_ids")) and chunk_in.has_raw(
                    chunk_in.offsets_key("time_group_ids", 0)
                )
                splitter_index_for_hit = self._build_splitter_value_index_per_hit(
                    n_rows=int(state["n_rows"]),
                    hit_offsets=hit_offsets,
                    hit_time_group_values=hit_time_group_values,
                    splitter_offsets=splitter_offsets,
                    splitter_time_group_offsets=(chunk_in.offsets("time_group_ids", 0) if has_splitter_tg else None),
                    splitter_time_group_values=(chunk_in.values("time_group_ids") if has_splitter_tg else None),
                )
                idx_sorted = splitter_index_for_hit[order]
                x_out[:, 4] = chunk_in.values("pred_hit_pion")[idx_sorted].astype(np.float32, copy=False)
                x_out[:, 5] = chunk_in.values("pred_hit_muon")[idx_sorted].astype(np.float32, copy=False)
                x_out[:, 6] = chunk_in.values("pred_hit_mip")[idx_sorted].astype(np.float32, copy=False)
            elif bool(getattr(owner, "include_targets", False)) and chunk_in.has_raw(chunk_in.values_key(self.particle_mask_field)):
                mask_sorted = chunk_in.values(self.particle_mask_field)[order]
                node_truth = self._particle_mask_to_multihot(mask_sorted)
                x_out[:, 4:7] = node_truth
                sorted_gid = global_group_id[order]
                for cls_id in range(3):
                    cls_mask = node_truth[:, cls_id] > 0.5
                    if np.any(cls_mask):
                        present = np.bincount(sorted_gid[cls_mask], minlength=total_graphs) > 0
                        group_truth_out[:, cls_id] = present.astype(np.float32, copy=False)
        else:
            coord_sorted = np.zeros((0,), dtype=np.float32)
            z_sorted = np.zeros((0,), dtype=np.float32)
            e_sorted = np.zeros((0,), dtype=np.float32)
            view_sorted = np.zeros((0,), dtype=np.int32)

        state["x_out"] = x_out
        state["tgroup_out"] = tgroup_out
        state["coord_sorted"] = coord_sorted
        state["z_sorted"] = z_sorted
        state["e_sorted"] = e_sorted
        state["view_sorted"] = view_sorted
        state["group_truth_out"] = group_truth_out
