from __future__ import annotations

from abc import abstractmethod
from collections.abc import Mapping
from typing import Any

from pioneerml.common.evaluation.metrics import compute_step_metrics

from ..base_pipeline_step import BasePipelineStep


class BaseEvaluationStep(BasePipelineStep):
    def default_metric_names(self) -> list[str]:
        return []

    def resolve_metric_names(self, cfg: Mapping[str, Any] | None = None) -> list[str]:
        raw = dict(cfg or self.get_config()).get("metrics")
        if raw is None:
            return list(self.default_metric_names())
        if isinstance(raw, (list, tuple)):
            names = [str(v) for v in raw if str(v).strip()]
            return names if names else list(self.default_metric_names())
        raise ValueError("evaluation.metrics must be a list of metric names when provided.")

    def apply_registered_metrics(
        self,
        *,
        metrics: dict[str, Any],
        context: Mapping[str, Any],
        metric_names: list[str] | None = None,
    ) -> dict[str, Any]:
        selected = list(metric_names) if metric_names is not None else self.resolve_metric_names()
        if not selected:
            return metrics
        metric_values = compute_step_metrics(metric_names=selected, context=context)
        metrics.update(metric_values)
        return metrics

    @abstractmethod
    def execute(self) -> dict:
        raise NotImplementedError
