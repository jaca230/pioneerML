from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from .base_metric import BaseMetric

STEP_METRIC_REGISTRY: dict[str, type[BaseMetric]] = {}


def register_step_metric(name: str):
    """Decorator to register a metric class for step-time metric composition."""

    def decorator(metric_cls: type[BaseMetric]) -> type[BaseMetric]:
        STEP_METRIC_REGISTRY[str(name)] = metric_cls
        return metric_cls

    return decorator


def create_step_metric(name: str) -> BaseMetric:
    cls = STEP_METRIC_REGISTRY.get(str(name))
    if cls is None:
        available = ", ".join(sorted(STEP_METRIC_REGISTRY))
        raise KeyError(f"Unknown metric '{name}'. Available metrics: [{available}]")
    return cls()


def compute_step_metrics(*, metric_names: Sequence[str], context: Mapping[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for metric_name in metric_names:
        metric = create_step_metric(metric_name)
        metric_values = metric.compute(context=context)
        for key, value in dict(metric_values).items():
            out[str(key)] = value
    return out
