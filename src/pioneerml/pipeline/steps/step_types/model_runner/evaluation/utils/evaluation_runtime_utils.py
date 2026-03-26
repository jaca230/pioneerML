from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from pioneerml.evaluation.metrics import MetricFactory
from pioneerml.evaluation.plots import PLOT_REGISTRY, PlotFactory


def apply_registered_metrics(
    *,
    metrics: dict[str, Any],
    context: Mapping[str, Any],
    metric_names: Sequence[str] | None,
) -> dict[str, Any]:
    selected = [str(name) for name in (metric_names or []) if str(name).strip()]
    if not selected:
        return metrics
    metric_values: dict[str, Any] = {}
    for metric_name in selected:
        metric = MetricFactory(metric_name=metric_name).build()
        values = metric.compute(context=context)
        for key, value in dict(values).items():
            metric_values[str(key)] = value
    metrics.update(metric_values)
    return metrics


def merge_plot_kwargs(
    *,
    plot_name: str,
    context: Mapping[str, Any] | None = None,
    default_plot_kwargs: Mapping[str, Any] | None = None,
    plot_kwargs_by_name: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    merged: dict[str, Any] = {}
    if isinstance(default_plot_kwargs, Mapping):
        merged.update(dict(default_plot_kwargs))
    if isinstance(plot_kwargs_by_name, Mapping):
        per_plot = plot_kwargs_by_name.get(plot_name)
        if isinstance(per_plot, Mapping):
            merged.update(dict(per_plot))
    if isinstance(context, Mapping):
        ctx_default = context.get("default_plot_kwargs")
        if isinstance(ctx_default, Mapping):
            merged.update(dict(ctx_default))
        ctx_by_name = context.get("plot_kwargs_by_name")
        if isinstance(ctx_by_name, Mapping):
            per_plot = ctx_by_name.get(plot_name)
            if isinstance(per_plot, Mapping):
                merged.update(dict(per_plot))
    return merged


def apply_registered_plots(
    *,
    plot_names: Sequence[str] | None,
    context: Mapping[str, Any] | None = None,
    default_plot_kwargs: Mapping[str, Any] | None = None,
    plot_kwargs_by_name: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    selected = [str(name) for name in (plot_names or []) if str(name).strip()]
    if not selected:
        return {}

    plot_outputs: dict[str, Any] = {}
    for plot_name in selected:
        if plot_name not in PLOT_REGISTRY:
            available = ", ".join(sorted(PLOT_REGISTRY))
            raise KeyError(f"Unknown plot '{plot_name}'. Available plots: [{available}]")

        kwargs = merge_plot_kwargs(
            plot_name=plot_name,
            context=context,
            default_plot_kwargs=default_plot_kwargs,
            plot_kwargs_by_name=plot_kwargs_by_name,
        )
        result = PlotFactory(plot_name=plot_name).build().render(**kwargs)
        if result is not None:
            key = f"{plot_name}_path" if isinstance(result, str) else f"{plot_name}_result"
            plot_outputs[key] = result
    return plot_outputs
