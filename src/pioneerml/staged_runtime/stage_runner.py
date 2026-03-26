from __future__ import annotations

from collections.abc import MutableMapping, Sequence
from typing import Any

from .base_stage import BaseStage
from .base_stage_context import BaseStageContext
from .stage_observers.base import StageObserver


class StageRunner:
    """Linear runner for a single phase stage sequence."""

    def __init__(
        self,
        *,
        stages: Sequence[BaseStage],
        observer: StageObserver | None = None,
        context_cls: type[BaseStageContext] = BaseStageContext,
    ) -> None:
        self.stages = tuple(stages)
        self.observer = observer or StageObserver()
        self.context_cls = context_cls

    def run(
        self,
        *,
        state: MutableMapping[str, Any],
        owner: Any,
        context_fields: dict[str, Any] | None = None,
    ) -> MutableMapping[str, Any]:
        runtime_context_fields = dict(context_fields or {})
        for i, stage in enumerate(self.stages):
            if bool(state.get("stop_pipeline", False)):
                break
            context = self.context_cls(
                stage_index=int(i),
                stage_name=str(stage.name),
                **runtime_context_fields,
            )
            self.observer.before_stage(context=context, state=state, owner=owner)
            try:
                stage.validate(state)
                stage.run(state=state, owner=owner)
            except Exception as exc:
                self.observer.on_error(context=context, state=state, owner=owner, error=exc)
                raise
            self.observer.after_stage(context=context, state=state, owner=owner)
        return state

