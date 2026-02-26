from __future__ import annotations

from collections.abc import MutableMapping
from typing import Any

import torch

from .base_stage import BaseStage


class BatchPackStage(BaseStage):
    """Base stage that packs chunk numpy arrays into torch tensors."""

    name = "pack_batch"
    requires = ("n_rows", "layout")
    provides = ("chunk_out",)

    def __init__(
        self,
        *,
        tensor_state_fields: dict[str, str] | None = None,
        tensor_layout_fields: dict[str, str] | None = None,
        scalar_state_fields: dict[str, str] | None = None,
        scalar_layout_fields: dict[str, str] | None = None,
        optional_tensor_state_fields: dict[str, str] | None = None,
        target_state_key: str = "targets_torch",
        target_output_key: str = "targets",
    ) -> None:
        self.tensor_state_fields = dict(
            tensor_state_fields
            or {
                "x": "x_out",
                "edge_index": "edge_index_out",
                "edge_attr": "edge_attr_out",
                "time_group_ids": "tgroup_out",
                "graph_event_ids": "graph_event_ids",
                "graph_group_ids": "graph_group_ids",
            }
        )
        self.tensor_layout_fields = dict(tensor_layout_fields or {"node_ptr": "node_ptr", "edge_ptr": "edge_ptr"})
        self.scalar_state_fields = dict(scalar_state_fields or {"num_rows": "n_rows"})
        self.scalar_layout_fields = dict(scalar_layout_fields or {"num_graphs": "total_graphs"})
        self.optional_tensor_state_fields = dict(optional_tensor_state_fields or {})
        self.target_state_key = str(target_state_key)
        self.target_output_key = str(target_output_key)

    @staticmethod
    def _as_torch(arr):
        if isinstance(arr, torch.Tensor):
            return arr
        return torch.from_numpy(arr)

    def run(self, *, state: MutableMapping[str, Any], loader) -> None:
        _ = loader
        layout = state["layout"]
        chunk_out: dict[str, Any] = {}

        for out_key, state_key in self.tensor_state_fields.items():
            if state_key not in state:
                raise RuntimeError(f"BatchPackStage missing tensor state key: {state_key}")
            chunk_out[out_key] = self._as_torch(state[state_key])
        for out_key, layout_key in self.tensor_layout_fields.items():
            if layout_key not in layout:
                raise RuntimeError(f"BatchPackStage missing tensor layout key: {layout_key}")
            chunk_out[out_key] = self._as_torch(layout[layout_key])
        for out_key, state_key in self.scalar_state_fields.items():
            if state_key not in state:
                raise RuntimeError(f"BatchPackStage missing scalar state key: {state_key}")
            chunk_out[out_key] = int(state[state_key])
        for out_key, layout_key in self.scalar_layout_fields.items():
            if layout_key not in layout:
                raise RuntimeError(f"BatchPackStage missing scalar layout key: {layout_key}")
            chunk_out[out_key] = int(layout[layout_key])
        for out_key, state_key in self.optional_tensor_state_fields.items():
            if state_key in state and state[state_key] is not None:
                chunk_out[out_key] = self._as_torch(state[state_key])

        targets_torch = state.get(self.target_state_key)
        if targets_torch is not None:
            chunk_out[self.target_output_key] = targets_torch
        state["chunk_out"] = chunk_out
