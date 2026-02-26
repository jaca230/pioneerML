from __future__ import annotations

from collections.abc import Iterable, MutableMapping
from typing import Any

from ..stage_context import StageContext
from .base import StageObserver


class CompositeStageObserver(StageObserver):
    """Fan-out observer wrapper."""

    def __init__(self, observers: Iterable[StageObserver]) -> None:
        self._observers = tuple(observers)

    def before_stage(self, *, context: StageContext, state: MutableMapping[str, Any], loader: Any) -> None:
        for obs in self._observers:
            obs.before_stage(context=context, state=state, loader=loader)

    def after_stage(self, *, context: StageContext, state: MutableMapping[str, Any], loader: Any) -> None:
        for obs in self._observers:
            obs.after_stage(context=context, state=state, loader=loader)

    def on_error(
        self,
        *,
        context: StageContext,
        state: MutableMapping[str, Any],
        loader: Any,
        error: Exception,
    ) -> None:
        for obs in self._observers:
            obs.on_error(context=context, state=state, loader=loader, error=error)

    def on_chunk_end(self, *, chunk_index: int, state: MutableMapping[str, Any], loader: Any) -> None:
        for obs in self._observers:
            obs.on_chunk_end(chunk_index=chunk_index, state=state, loader=loader)
