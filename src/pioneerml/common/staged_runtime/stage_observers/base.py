from __future__ import annotations

from abc import ABC
from collections.abc import MutableMapping
from typing import Any

from ..base_stage_context import BaseStageContext


class StageObserver(ABC):
    """Observer interface for stage-run diagnostics."""

    def before_stage(self, *, context: BaseStageContext, state: MutableMapping[str, Any], owner: Any) -> None:
        _ = context
        _ = state
        _ = owner

    def after_stage(self, *, context: BaseStageContext, state: MutableMapping[str, Any], owner: Any) -> None:
        _ = context
        _ = state
        _ = owner

    def on_error(
        self,
        *,
        context: BaseStageContext,
        state: MutableMapping[str, Any],
        owner: Any,
        error: Exception,
    ) -> None:
        _ = context
        _ = state
        _ = owner
        _ = error

    def on_run_end(
        self,
        *,
        run_index: int,
        phase: str,
        state: MutableMapping[str, Any],
        owner: Any,
    ) -> None:
        _ = run_index
        _ = phase
        _ = state
        _ = owner
