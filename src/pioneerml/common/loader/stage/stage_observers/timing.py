from __future__ import annotations

import time
from collections.abc import MutableMapping
from typing import Any

from ..stage_context import StageContext
from ..utils.loader_diagnostics import LoaderDiagnostics
from .base import StageObserver


class TimingObserver(StageObserver):
    """Per-stage elapsed-time observer."""

    def __init__(self, *, diagnostics: LoaderDiagnostics) -> None:
        self.diagnostics = diagnostics
        self._start_ns: dict[tuple[int, int], int] = {}

    def before_stage(self, *, context: StageContext, state: MutableMapping[str, Any], loader: Any) -> None:
        _ = state
        _ = loader
        self._start_ns[(context.chunk_index, context.stage_index)] = time.perf_counter_ns()

    def after_stage(self, *, context: StageContext, state: MutableMapping[str, Any], loader: Any) -> None:
        _ = state
        _ = loader
        key = (context.chunk_index, context.stage_index)
        t0 = self._start_ns.pop(key, None)
        if t0 is None:
            return
        ms = (time.perf_counter_ns() - t0) / 1_000_000.0
        self.diagnostics.record_stage_ms(stage_name=context.stage_name, elapsed_ms=ms)

    def on_error(
        self,
        *,
        context: StageContext,
        state: MutableMapping[str, Any],
        loader: Any,
        error: Exception,
    ) -> None:
        _ = state
        _ = loader
        _ = error
        self.diagnostics.record_stage_error(stage_name=context.stage_name)
