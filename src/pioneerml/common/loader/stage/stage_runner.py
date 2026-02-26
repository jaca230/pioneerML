from __future__ import annotations

from collections.abc import MutableMapping, Sequence
from typing import Any

from .stage_context import StageContext
from .stage_observers.base import StageObserver
from .stages import BaseStage


class StageRunner:
    """Linear stage runner with observer hooks."""

    def __init__(self, *, stages: Sequence[BaseStage], observer: StageObserver | None = None) -> None:
        self.stages = tuple(stages)
        self.observer = observer or StageObserver()

    def run(
        self,
        *,
        state: MutableMapping[str, Any],
        loader: Any,
        chunk_index: int,
        raw_num_rows: int,
    ) -> MutableMapping[str, Any]:
        for i, stage in enumerate(self.stages):
            if bool(state.get("stop_pipeline", False)):
                break
            context = StageContext(
                chunk_index=int(chunk_index),
                stage_index=int(i),
                stage_name=str(stage.name),
                raw_num_rows=int(raw_num_rows),
            )
            self.observer.before_stage(context=context, state=state, loader=loader)
            try:
                stage.validate(state)
                stage.run(state=state, loader=loader)
            except Exception as exc:
                self.observer.on_error(context=context, state=state, loader=loader, error=exc)
                raise
            self.observer.after_stage(context=context, state=state, loader=loader)
        self.observer.on_chunk_end(chunk_index=int(chunk_index), state=state, loader=loader)
        return state
