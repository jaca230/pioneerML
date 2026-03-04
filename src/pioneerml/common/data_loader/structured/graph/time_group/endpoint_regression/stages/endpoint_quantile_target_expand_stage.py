from __future__ import annotations

from collections.abc import MutableMapping
from typing import Any

import torch

from pioneerml.common.data_loader.stage.stages import BaseStage


class EndpointQuantileTargetExpandStage(BaseStage):
    """Expand base endpoint targets `[6] -> [18]` by repeating each coordinate per quantile slot."""

    name = "expand_quantile_targets"
    requires = ("y_graph",)
    provides = ("y_graph",)

    def __init__(self, *, repeats: int = 3, target_key: str = "y_graph") -> None:
        self.repeats = int(repeats)
        self.target_key = str(target_key)

    def run_loader(self, *, state: MutableMapping[str, Any], owner) -> None:
        _ = owner
        y_graph = state.get(self.target_key)
        if y_graph is None:
            return
        if not isinstance(y_graph, torch.Tensor):
            raise RuntimeError(f"Expected tensor at state['{self.target_key}'], got {type(y_graph).__name__}.")
        state[self.target_key] = y_graph.repeat_interleave(self.repeats, dim=1)
