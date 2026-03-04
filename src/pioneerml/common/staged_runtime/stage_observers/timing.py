from __future__ import annotations

import time
from collections.abc import MutableMapping
from typing import Any

from ..base_diagnostics import BaseDiagnostics
from ..base_stage_context import BaseStageContext
from .base import StageObserver


class TimingObserver(StageObserver):
    """Per-stage elapsed-time observer."""

    def __init__(self, *, diagnostics: BaseDiagnostics) -> None:
        self.diagnostics = diagnostics
        self._start_ns: dict[int, int] = {}

    def before_stage(self, *, context: BaseStageContext, state: MutableMapping[str, Any], owner: Any) -> None:
        _ = state
        _ = owner
        self._start_ns[id(context)] = time.perf_counter_ns()

    def after_stage(self, *, context: BaseStageContext, state: MutableMapping[str, Any], owner: Any) -> None:
        _ = state
        _ = owner
        t0 = self._start_ns.pop(id(context), None)
        if t0 is None:
            return
        ms = (time.perf_counter_ns() - t0) / 1_000_000.0
        self.diagnostics.record_stage_ms(stage_name=context.stage_name, elapsed_ms=ms)

    def on_error(
        self,
        *,
        context: BaseStageContext,
        state: MutableMapping[str, Any],
        owner: Any,
        error: Exception,
    ) -> None:
        _ = state
        _ = owner
        _ = error
        self.diagnostics.record_stage_error(stage_name=context.stage_name)
