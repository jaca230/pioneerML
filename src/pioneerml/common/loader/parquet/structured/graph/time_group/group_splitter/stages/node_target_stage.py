from __future__ import annotations

from collections.abc import MutableMapping
from typing import Any

import numpy as np
import torch

from pioneerml.common.loader.array_store import NDArrayStore
from pioneerml.common.loader.stage.stages import NodeTargetStage as BaseNodeTargetStage


class NodeTargetStage(BaseNodeTargetStage):
    """Build per-node multi-hot targets from hits_particle_mask."""

    name = "build_node_targets"
    requires = ("layout",)
    provides = ("y_node", "group_truth_out")

    def __init__(
        self,
        *,
        input_state_key: str = "features_in",
        particle_mask_field: str = "hits_particle_mask",
        num_classes: int = 3,
    ) -> None:
        self.input_state_key = str(input_state_key)
        self.particle_mask_field = str(particle_mask_field)
        self.num_classes = int(num_classes)

    @staticmethod
    def _particle_mask_to_multihot(mask_values: np.ndarray) -> np.ndarray:
        out = np.zeros((mask_values.size, 3), dtype=np.float32)
        out[:, 0] = ((mask_values & 1) != 0).astype(np.float32, copy=False)
        out[:, 1] = ((mask_values & 2) != 0).astype(np.float32, copy=False)
        out[:, 2] = ((mask_values & 4) != 0).astype(np.float32, copy=False)
        return out

    def run(self, *, state: MutableMapping[str, Any], loader) -> None:
        layout = state["layout"]
        total_nodes = int(layout["total_nodes"])
        total_graphs = int(layout["total_graphs"])
        include_targets = bool(getattr(loader, "include_targets", False))

        if not include_targets:
            state["y_node"] = None
            state["group_truth_out"] = None
            return

        chunk_in = state.get(self.input_state_key)
        if not isinstance(chunk_in, NDArrayStore):
            raise RuntimeError(f"Stage '{self.name}' missing required input state map: {self.input_state_key}")

        global_group_id = layout["global_group_id"]
        if total_nodes > 0:
            order = np.argsort(global_group_id, kind="stable")
            mask_sorted = chunk_in.values(self.particle_mask_field)[order]
            node_targets = self._particle_mask_to_multihot(mask_sorted)
            sorted_gid = global_group_id[order]
        else:
            node_targets = np.zeros((0, self.num_classes), dtype=np.float32)
            sorted_gid = np.zeros((0,), dtype=np.int64)

        group_truth = np.zeros((total_graphs, self.num_classes), dtype=np.float32)
        for cls_id in range(self.num_classes):
            mask = node_targets[:, cls_id] > 0.5
            if not np.any(mask):
                continue
            present = np.bincount(sorted_gid[mask], minlength=total_graphs) > 0
            group_truth[:, cls_id] = present.astype(np.float32, copy=False)

        state["y_node"] = torch.from_numpy(node_targets)
        state["group_truth_out"] = group_truth
