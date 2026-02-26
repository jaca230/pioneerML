from __future__ import annotations

from abc import ABC
from collections.abc import MutableMapping
from typing import Any

from ..stage_context import StageContext


class StageObserver(ABC):
    """Observer interface for stage-run diagnostics."""

    def before_stage(self, *, context: StageContext, state: MutableMapping[str, Any], loader: Any) -> None:
        _ = context
        _ = state
        _ = loader

    def after_stage(self, *, context: StageContext, state: MutableMapping[str, Any], loader: Any) -> None:
        _ = context
        _ = state
        _ = loader

    def on_error(
        self,
        *,
        context: StageContext,
        state: MutableMapping[str, Any],
        loader: Any,
        error: Exception,
    ) -> None:
        _ = context
        _ = state
        _ = loader
        _ = error

    def on_chunk_end(self, *, chunk_index: int, state: MutableMapping[str, Any], loader: Any) -> None:
        _ = chunk_index
        _ = state
        _ = loader
