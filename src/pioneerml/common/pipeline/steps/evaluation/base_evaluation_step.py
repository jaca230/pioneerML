from __future__ import annotations

from abc import abstractmethod
from collections.abc import Mapping
from typing import Any

from pioneerml.common.evaluation.metrics import compute_step_metrics
from pioneerml.common.evaluation.plots import PLOT_CLASSES

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

    def default_plot_names(self) -> list[str]:
        return []

    def resolve_plot_names(self, cfg: Mapping[str, Any] | None = None) -> list[str]:
        raw = dict(cfg or self.get_config()).get("plots")
        if raw is None:
            return list(self.default_plot_names())
        if isinstance(raw, (list, tuple)):
            names = [str(v) for v in raw if str(v).strip()]
            return names if names else list(self.default_plot_names())
        raise ValueError("evaluation.plots must be a list of plot names when provided.")

    def apply_registered_plots(
        self,
        *,
        context: Mapping[str, Any],
        plot_names: list[str] | None = None,
    ) -> dict[str, Any]:
        selected = list(plot_names) if plot_names is not None else self.resolve_plot_names()
        if not selected:
            return {}

        plot_outputs: dict[str, Any] = {}
        default_plot_kwargs = context.get("default_plot_kwargs")
        plot_kwargs_by_name = context.get("plot_kwargs_by_name")

        for plot_name in selected:
            plot_cls = PLOT_CLASSES.get(plot_name)
            if plot_cls is None:
                available = ", ".join(sorted(PLOT_CLASSES))
                raise KeyError(f"Unknown plot '{plot_name}'. Available plots: [{available}]")

            kwargs: dict[str, Any] = {}
            if isinstance(default_plot_kwargs, Mapping):
                kwargs.update(dict(default_plot_kwargs))
            if isinstance(plot_kwargs_by_name, Mapping):
                per_plot = plot_kwargs_by_name.get(plot_name)
                if isinstance(per_plot, Mapping):
                    kwargs.update(dict(per_plot))

            result = plot_cls().render(**kwargs)
            if result is not None:
                key = f"{plot_name}_path" if isinstance(result, str) else f"{plot_name}_result"
                plot_outputs[key] = result

        return plot_outputs

    @abstractmethod
    def execute(self) -> dict:
        raise NotImplementedError
