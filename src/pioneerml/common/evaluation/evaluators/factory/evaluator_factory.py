from __future__ import annotations

from typing import Any

from ..base_evaluator import BaseEvaluator
from .registry import resolve_evaluator


class EvaluatorFactory:
    def __init__(
        self,
        *,
        evaluator_cls: type[BaseEvaluator] | None = None,
        evaluator_name: str | None = None,
    ) -> None:
        if evaluator_cls is None and evaluator_name is None:
            raise ValueError("EvaluatorFactory requires either evaluator_cls or evaluator_name.")
        self.evaluator_cls = evaluator_cls
        self.evaluator_name = None if evaluator_name is None else str(evaluator_name).strip()

    def _resolve_evaluator_class(self) -> type[BaseEvaluator]:
        if self.evaluator_cls is not None:
            return self.evaluator_cls
        if self.evaluator_name is None:
            raise RuntimeError("EvaluatorFactory has neither evaluator_cls nor evaluator_name.")
        return resolve_evaluator(self.evaluator_name)

    def build_evaluator(self, *, evaluator_params: dict[str, Any] | None = None) -> BaseEvaluator:
        evaluator_cls = self._resolve_evaluator_class()
        params = dict(evaluator_params or {})
        if hasattr(evaluator_cls, "from_factory"):
            return evaluator_cls.from_factory(config=params)
        return evaluator_cls(**params)
