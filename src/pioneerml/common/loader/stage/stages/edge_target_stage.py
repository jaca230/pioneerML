from __future__ import annotations

from collections.abc import MutableMapping
from typing import Any

from .base_target_stage import BaseTargetStage


class EdgeTargetStage(BaseTargetStage):
    """Base stage for edge-level target construction."""

    name = "build_edge_targets"
    requires = ("layout", "edge_index_out")

    def run(self, *, state: MutableMapping[str, Any], loader) -> None:
        raise NotImplementedError("EdgeTargetStage subclasses must implement run().")
