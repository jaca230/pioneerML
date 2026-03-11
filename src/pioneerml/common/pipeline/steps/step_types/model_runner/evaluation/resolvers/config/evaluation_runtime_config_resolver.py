from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from .......resolver import BaseConfigResolver


class EvaluationRuntimeConfigResolver(BaseConfigResolver):
    def resolve(self, *, cfg: dict[str, Any]) -> None:
        cfg["metric_names"] = self.resolve_metric_names(cfg=cfg)
        cfg["plot_names"] = self.resolve_plot_names(cfg=cfg)

    def resolve_metric_names(self, *, cfg: Mapping[str, Any]) -> list[str]:
        default_names = list(getattr(self.step, "default_metric_names")())
        return self.resolve_registered_names(
            cfg=cfg,
            config_key="metrics",
            default_names=default_names,
            error_context="evaluation.metrics",
        )

    def resolve_plot_names(self, *, cfg: Mapping[str, Any]) -> list[str]:
        default_names = list(getattr(self.step, "default_plot_names")())
        return self.resolve_registered_names(
            cfg=cfg,
            config_key="plots",
            default_names=default_names,
            error_context="evaluation.plots",
        )

    @staticmethod
    def resolve_registered_names(
        *,
        cfg: Mapping[str, Any],
        config_key: str,
        default_names: list[str],
        error_context: str,
    ) -> list[str]:
        raw = dict(cfg).get(config_key)
        if raw is None:
            return list(default_names)
        if isinstance(raw, (list, tuple)):
            names = [str(value) for value in raw if str(value).strip()]
            return names if names else list(default_names)
        raise ValueError(f"{error_context} must be a list of names when provided.")

    @staticmethod
    def merge_plot_kwargs(
        *,
        context: Mapping[str, Any],
        plot_name: str,
    ) -> dict[str, Any]:
        kwargs: dict[str, Any] = {}
        default_plot_kwargs = context.get("default_plot_kwargs")
        plot_kwargs_by_name = context.get("plot_kwargs_by_name")
        if isinstance(default_plot_kwargs, Mapping):
            kwargs.update(dict(default_plot_kwargs))
        if isinstance(plot_kwargs_by_name, Mapping):
            per_plot = plot_kwargs_by_name.get(plot_name)
            if isinstance(per_plot, Mapping):
                kwargs.update(dict(per_plot))
        return kwargs
