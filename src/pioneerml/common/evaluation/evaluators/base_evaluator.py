from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from pioneerml.common.evaluation.metrics import BaseMetric, MetricFactory
from pioneerml.common.evaluation.plots import BasePlot, PlotFactory


class BaseEvaluator(ABC):
    metric_base_class = BaseMetric
    plot_base_class = BasePlot
    default_metric_names: tuple[str, ...] = ()
    default_plot_names: tuple[str, ...] = ()

    @classmethod
    def from_factory(cls, *, config: Mapping[str, Any] | None = None) -> "BaseEvaluator":
        _ = config
        return cls()

    def resolve_plot_path(self, config: dict | None) -> str | None:
        if not config:
            return None
        if config.get("plot_path"):
            return str(config["plot_path"])
        if config.get("plot_dir"):
            plot_dir = Path(str(config["plot_dir"]))
            plot_dir.mkdir(parents=True, exist_ok=True)
            return str(plot_dir / "loss_curves.png")
        return None

    def resolve_metric_names(self, *, config: Mapping[str, Any]) -> list[str]:
        raw = config.get("metrics")
        if raw is None:
            return list(self.default_metric_names)
        if not isinstance(raw, (list, tuple)):
            raise ValueError("evaluation.metrics must be a list when provided.")
        names = [str(name) for name in raw if str(name).strip()]
        return names if names else list(self.default_metric_names)

    def resolve_plot_names(self, *, config: Mapping[str, Any]) -> list[str]:
        raw = config.get("plots")
        if raw is None:
            return list(self.default_plot_names)
        if not isinstance(raw, (list, tuple)):
            raise ValueError("evaluation.plots must be a list when provided.")
        names = [str(name) for name in raw if str(name).strip()]
        return names if names else list(self.default_plot_names)

    @staticmethod
    def _resolve_default_plot_kwargs(*, config: Mapping[str, Any]) -> dict[str, Any]:
        raw = config.get("default_plot_kwargs")
        if raw is None:
            return {}
        if not isinstance(raw, Mapping):
            raise ValueError("evaluation.default_plot_kwargs must be a mapping when provided.")
        return dict(raw)

    @staticmethod
    def _resolve_plot_kwargs_by_name(*, config: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
        raw = config.get("plot_kwargs_by_name")
        if raw is None:
            return {}
        if not isinstance(raw, Mapping):
            raise ValueError("evaluation.plot_kwargs_by_name must be a mapping when provided.")
        out: dict[str, dict[str, Any]] = {}
        for name, kwargs in raw.items():
            if kwargs is None:
                continue
            if not isinstance(kwargs, Mapping):
                raise ValueError("evaluation.plot_kwargs_by_name values must be mappings.")
            out[str(name)] = dict(kwargs)
        return out

    @abstractmethod
    def build_context(self, *, module, loader, config: Mapping[str, Any]) -> dict[str, Any]:
        raise NotImplementedError

    def metric_context(self, *, context: Mapping[str, Any], config: Mapping[str, Any]) -> Mapping[str, Any]:
        metric_ctx = context.get("metric_context")
        if isinstance(metric_ctx, Mapping):
            return metric_ctx
        return context

    def plot_context_by_name(self, *, context: Mapping[str, Any], config: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
        by_name: dict[str, dict[str, Any]] = {}
        default_kwargs = self._resolve_default_plot_kwargs(config=config)
        configured_by_name = self._resolve_plot_kwargs_by_name(config=config)

        context_by_name = context.get("plot_kwargs_by_name")
        if isinstance(context_by_name, Mapping):
            for name, kwargs in context_by_name.items():
                if isinstance(kwargs, Mapping):
                    by_name[str(name)] = dict(kwargs)

        for name, kwargs in configured_by_name.items():
            if name in by_name:
                by_name[name].update(kwargs)
            else:
                by_name[name] = dict(kwargs)

        if default_kwargs:
            for name in set(by_name):
                merged = dict(default_kwargs)
                merged.update(by_name[name])
                by_name[name] = merged
        return by_name

    def finalize_results(
        self,
        *,
        results: dict[str, Any],
        context: Mapping[str, Any],
        config: Mapping[str, Any],
    ) -> dict[str, Any]:
        return results

    def evaluate(self, *, module, loader, config: dict[str, Any] | None = None) -> dict[str, Any]:
        cfg = dict(config or {})
        context = self.build_context(module=module, loader=loader, config=cfg)
        results: dict[str, Any] = {}
        base_metrics = context.get("base_metrics")
        if isinstance(base_metrics, Mapping):
            results.update(dict(base_metrics))

        metric_names = self.resolve_metric_names(config=cfg)
        if metric_names:
            results.update(
                self.compute_metrics(
                    metric_names=metric_names,
                    context=self.metric_context(context=context, config=cfg),
                )
            )

        plot_names = self.resolve_plot_names(config=cfg)
        if plot_names:
            context_by_plot = self.plot_context_by_name(context=context, config=cfg)
            results.update(self.render_plots(plot_names=plot_names, context_by_plot=context_by_plot))

        return self.finalize_results(results=results, context=context, config=cfg)

    @staticmethod
    def compute_metrics(*, metric_names: Sequence[str], context: Mapping[str, Any]) -> dict[str, Any]:
        out: dict[str, Any] = {}
        for metric_name in metric_names:
            metric = MetricFactory(metric_name=str(metric_name)).build()
            metric_values = metric.compute(context=context)
            for key, value in dict(metric_values).items():
                out[str(key)] = value
        return out

    @staticmethod
    def render_plots(
        *,
        plot_names: Sequence[str],
        context_by_plot: Mapping[str, Mapping[str, Any]] | None = None,
    ) -> dict[str, Any]:
        outputs: dict[str, Any] = {}
        by_plot = dict(context_by_plot or {})
        for name in plot_names:
            plot = PlotFactory(plot_name=str(name)).build()
            kwargs = dict(by_plot.get(str(name), {}))
            result = plot.render(**kwargs)
            if result is None:
                continue
            key = f"{name}_path" if isinstance(result, str) else f"{name}_result"
            outputs[key] = result
        return outputs
