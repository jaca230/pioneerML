from __future__ import annotations

from collections.abc import MutableMapping
from typing import Any

import numpy as np

from ...array_store.ndarray_store import NDArrayStore
from .base_stage import BaseStage


class NodeFeatureStage(BaseStage):
    """Base stage that builds node-level features and sorted edge inputs."""

    name = "build_nodes"
    requires = ("layout",)
    provides = ("x_out", "tgroup_out", "coord_sorted", "z_sorted", "e_sorted", "view_sorted")

    def __init__(
        self,
        *,
        input_state_key: str = "chunk_in",
        coord_field: str = "hits_coord",
        z_field: str = "hits_z",
        edep_field: str = "hits_edep",
        strip_type_field: str = "hits_strip_type",
        time_group_field: str = "hits_time_group",
        node_feature_dim: int = 4,
    ) -> None:
        self.input_state_key = str(input_state_key)
        self.coord_field = str(coord_field)
        self.z_field = str(z_field)
        self.edep_field = str(edep_field)
        self.strip_type_field = str(strip_type_field)
        self.time_group_field = str(time_group_field)
        self.node_feature_dim = int(node_feature_dim)

    def run(self, *, state: MutableMapping[str, Any], loader) -> None:
        _ = loader
        layout = state["layout"]
        chunk_in = state.get(self.input_state_key)
        if not isinstance(chunk_in, NDArrayStore):
            raise RuntimeError(
                f"Stage '{self.name}' missing required input state map: {self.input_state_key}"
            )
        total_nodes = int(layout["total_nodes"])

        x_out = np.empty((total_nodes, self.node_feature_dim), dtype=np.float32)
        tgroup_out = np.empty((total_nodes,), dtype=np.int64)

        if total_nodes > 0:
            order = np.argsort(layout["global_group_id"], kind="stable")
            coord_sorted = chunk_in.values(self.coord_field)[order]
            z_sorted = chunk_in.values(self.z_field)[order]
            e_sorted = chunk_in.values(self.edep_field)[order]
            view_sorted = chunk_in.values(self.strip_type_field)[order]

            x_out[:, 0] = coord_sorted
            x_out[:, 1] = z_sorted
            x_out[:, 2] = e_sorted
            x_out[:, 3] = view_sorted.astype(np.float32, copy=False)
            tgroup_out[:] = chunk_in.values(self.time_group_field)[order]
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
