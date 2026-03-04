from __future__ import annotations

from collections.abc import MutableMapping, Sequence
from typing import Any

from pioneerml.common.staged_runtime import PhaseRunner, StageRunner
from pioneerml.common.staged_runtime.stage_observers import StageObserver

from .stage_context import WriterStageContext
from .stages.base_stage import BaseWriterStage


class WriterStageRunner:
    def __init__(
        self,
        *,
        init_stages: Sequence[BaseWriterStage],
        chunk_stages: Sequence[BaseWriterStage],
        finalize_stages: Sequence[BaseWriterStage],
        observer: StageObserver | None = None,
    ) -> None:
        self._phase_runner = PhaseRunner()
        self._phase_runner.register_phase(
            name="start",
            runner=StageRunner(
                stages=tuple(init_stages),
                observer=observer,
                context_cls=WriterStageContext,
            ),
        )
        self._phase_runner.register_phase(
            name="chunk",
            runner=StageRunner(
                stages=tuple(chunk_stages),
                observer=observer,
                context_cls=WriterStageContext,
            ),
        )
        self._phase_runner.register_phase(
            name="finalize",
            runner=StageRunner(
                stages=tuple(finalize_stages),
                observer=observer,
                context_cls=WriterStageContext,
            ),
        )

    @staticmethod
    def _context_fields(state: MutableMapping[str, Any]) -> dict[str, Any]:
        chunk_index = int(state.get("chunk_index", 0))
        raw_num_rows = int(state.get("raw_num_rows", state.get("num_rows", 0)))
        return {
            "chunk_index": chunk_index,
            "raw_num_rows": raw_num_rows,
        }

    def run_phase(self, *, phase: str, state: MutableMapping[str, Any], owner) -> MutableMapping[str, Any]:
        return self._phase_runner.run_phase(
            name=str(phase),
            state=state,
            owner=owner,
            context_fields=self._context_fields(state),
        )
