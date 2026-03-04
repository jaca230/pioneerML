from __future__ import annotations

from collections.abc import MutableMapping
from typing import Any

from .base_target_stage import BaseTargetStage


class NodeTargetStage(BaseTargetStage):
    """Base stage for node-level target construction."""

    name = "build_node_targets"
    requires = ("layout",)

    def run_loader(self, *, state: MutableMapping[str, Any], owner) -> None:
        raise NotImplementedError("NodeTargetStage subclasses must implement run_loader().")
