from __future__ import annotations

from collections.abc import Iterable, MutableMapping
from typing import Any

from ..base_stage_context import BaseStageContext
from .base import StageObserver


class CompositeStageObserver(StageObserver):
    """Fan-out observer wrapper."""

    def __init__(self, observers: Iterable[StageObserver]) -> None:
        self._observers = tuple(observers)

    def before_stage(self, *, context: BaseStageContext, state: MutableMapping[str, Any], owner: Any) -> None:
        for obs in self._observers:
            obs.before_stage(context=context, state=state, owner=owner)

    def after_stage(self, *, context: BaseStageContext, state: MutableMapping[str, Any], owner: Any) -> None:
        for obs in self._observers:
            obs.after_stage(context=context, state=state, owner=owner)

    def on_error(
        self,
        *,
        context: BaseStageContext,
        state: MutableMapping[str, Any],
        owner: Any,
        error: Exception,
    ) -> None:
        for obs in self._observers:
            obs.on_error(context=context, state=state, owner=owner, error=error)

    def on_run_end(self, *, run_index: int, phase: str, state: MutableMapping[str, Any], owner: Any) -> None:
        for obs in self._observers:
            obs.on_run_end(run_index=run_index, phase=phase, state=state, owner=owner)
