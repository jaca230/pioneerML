from __future__ import annotations

from collections.abc import MutableMapping
from typing import Any

from .base_stage import BaseStage


class BaseTargetStage(BaseStage):
    """Base stage for target construction."""

    provides: tuple[str, ...] = ()

    @staticmethod
    def include_targets(*, loader, state: MutableMapping[str, Any]) -> bool:
        _ = state
        return bool(getattr(loader, "include_targets", False))
