from __future__ import annotations

from collections.abc import MutableMapping
from typing import Any

import numpy as np

from pioneerml.common.data_loader.loaders.array_store import NDArrayStore
from pioneerml.common.data_loader.loaders.stage.stages import BaseStage


class EventNodeFeatureStage(BaseStage):
    """Build event-splitter node features and optional target fallbacks."""

    name = "build_nodes"
    requires = ("layout",)
    provides = (
        "x_out",
        "time_group_ids_out",
        "coord_values",
        "z_values",
        "edep_values",
        "view_values",
        "node_truth_out",
        "group_truth_out",
    )

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
        node_feature_dim: int = 4,
        num_classes: int = 3,
    ) -> None:
        self.input_state_key = str(input_state_key)
        self.coord_field = str(coord_field)
        self.z_field = str(z_field)
        self.edep_field = str(edep_field)
        self.strip_type_field = str(strip_type_field)
        self.time_group_field = str(time_group_field)
        self.particle_mask_field = str(particle_mask_field)
        self.node_feature_dim = int(node_feature_dim)
        self.num_classes = int(num_classes)

    @staticmethod
    def _particle_mask_to_multihot(mask_values: np.ndarray) -> np.ndarray:
        out = np.zeros((mask_values.size, 3), dtype=np.float32)
        out[:, 0] = ((mask_values & 1) != 0).astype(np.float32, copy=False)
        out[:, 1] = ((mask_values & 2) != 0).astype(np.float32, copy=False)
        out[:, 2] = ((mask_values & 4) != 0).astype(np.float32, copy=False)
        return out

    def run_loader(self, *, state: MutableMapping[str, Any], owner) -> None:
        _ = owner
        layout = state["layout"]
        chunk_in = state.get(self.input_state_key)
        if not isinstance(chunk_in, NDArrayStore):
            raise RuntimeError(
                f"Stage '{self.name}' missing required input state map: {self.input_state_key}"
            )

        total_nodes = int(layout["total_nodes"])
        total_groups = int(layout["total_groups"])
        hit_counts = np.asarray(layout["hit_counts"], dtype=np.int64)
        global_group_id = np.asarray(layout["global_group_id"], dtype=np.int64)

        coord_values = chunk_in.values(self.coord_field)
        z_values = chunk_in.values(self.z_field)
        edep_values = chunk_in.values(self.edep_field)
        view_values = chunk_in.values(self.strip_type_field).astype(np.int32, copy=False)
        time_group_ids_out = chunk_in.values(self.time_group_field).astype(np.int64, copy=False)

        x_out = np.empty((total_nodes, self.node_feature_dim), dtype=np.float32)
        x_out[:, 0] = coord_values
        x_out[:, 1] = z_values
        x_out[:, 2] = edep_values
        x_out[:, 3] = view_values.astype(np.float32, copy=False)

        node_truth_out = np.zeros((total_nodes, self.num_classes), dtype=np.float32)
        group_truth_out = np.zeros((total_groups, self.num_classes), dtype=np.float32)

        if chunk_in.has_raw(chunk_in.values_key(self.particle_mask_field)):
            mask_offsets = chunk_in.offsets(self.particle_mask_field, 0)
            mask_counts = (mask_offsets[1:] - mask_offsets[:-1]).astype(np.int64, copy=False)
            if not np.array_equal(mask_counts, hit_counts):
                raise RuntimeError("hits_particle_mask list lengths do not match event hit counts.")
            particle_mask_values = chunk_in.values(self.particle_mask_field).astype(np.int32, copy=False)
            node_truth_out = self._particle_mask_to_multihot(particle_mask_values)

            if total_groups > 0:
                for cls_id in range(self.num_classes):
                    cls_mask = node_truth_out[:, cls_id] > 0.5
                    if np.any(cls_mask):
                        present = np.bincount(global_group_id[cls_mask], minlength=total_groups) > 0
                        group_truth_out[:, cls_id] = present.astype(np.float32, copy=False)

        state["x_out"] = x_out
        state["time_group_ids_out"] = time_group_ids_out
        state["coord_values"] = coord_values.astype(np.float32, copy=False)
        state["z_values"] = z_values.astype(np.float32, copy=False)
        state["edep_values"] = edep_values.astype(np.float32, copy=False)
        state["view_values"] = view_values
        state["node_truth_out"] = node_truth_out
        state["group_truth_out"] = group_truth_out
