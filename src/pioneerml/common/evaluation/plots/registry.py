from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from .base_plot import BasePlot

STEP_PLOT_REGISTRY: dict[str, type[BasePlot]] = {}


def register_step_plot(name: str):
    def _decorator(plot_cls: type[BasePlot]) -> type[BasePlot]:
        STEP_PLOT_REGISTRY[str(name)] = plot_cls
        return plot_cls

    return _decorator


def create_step_plot(name: str, **kwargs: Any) -> BasePlot:
    cls = STEP_PLOT_REGISTRY.get(str(name))
    if cls is None:
        available = ", ".join(sorted(STEP_PLOT_REGISTRY))
        raise KeyError(f"Unknown plot '{name}'. Available plots: [{available}]")
    return cls(**kwargs)


def render_step_plots(
    *,
    plot_names: Sequence[str],
    context_by_plot: Mapping[str, Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    outputs: dict[str, Any] = {}
    by_plot = dict(context_by_plot or {})
    for name in plot_names:
        plot = create_step_plot(name)
        kwargs = dict(by_plot.get(str(name), {}))
        result = plot.render(**kwargs)
        if result is None:
            continue
        key = f"{name}_path" if isinstance(result, str) else f"{name}_result"
        outputs[key] = result
    return outputs
