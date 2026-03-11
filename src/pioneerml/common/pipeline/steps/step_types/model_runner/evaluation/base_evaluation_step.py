from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from pioneerml.common.evaluation.metrics import compute_step_metrics
from pioneerml.common.evaluation.plots import STEP_PLOT_REGISTRY, create_step_plot
from .payloads import EvaluationStepPayload
from .resolvers import EvaluationRuntimeConfigResolver, EvaluationRuntimeStateResolver
from .utils import build_evaluation_loader_bundle

from ..base_pipeline_step import BasePipelineStep
from pioneerml.common.data_loader import LoaderFactory


class BaseEvaluationStep(BasePipelineStep):
    DEFAULT_CONFIG = {
        "metrics": [],
        "plots": [],
        "loader_config": {
            "base": {
                "batch_size": 1,
                "mode": "train",
                "chunk_row_groups": 4,
                "chunk_workers": None,
                "sample_fraction": 1.0,
                "train_fraction": 0.80,
                "val_fraction": 0.10,
                "test_fraction": 0.10,
                "split_seed": None,
            },
            "evaluate": {"mode": "train", "shuffle_batches": False, "log_diagnostics": False},
            "val": {"mode": "train", "shuffle_batches": False, "log_diagnostics": False},
        },
    }
    config_resolver_classes = (EvaluationRuntimeConfigResolver,)
    payload_resolver_classes = (EvaluationRuntimeStateResolver,)

    def default_metric_names(self) -> list[str]:
        return []

    def resolve_metric_names(self, cfg: Mapping[str, Any] | None = None) -> list[str]:
        src = dict(cfg or self.config_json)
        names = src.get("metric_names")
        if isinstance(names, list):
            return [str(v) for v in names]
        return []

    def apply_registered_metrics(
        self,
        *,
        metrics: dict[str, Any],
        context: Mapping[str, Any],
        metric_names: list[str] | None = None,
    ) -> dict[str, Any]:
        selected = list(metric_names) if metric_names is not None else list(self.config_json.get("metric_names") or [])
        if not selected:
            return metrics
        metric_values = compute_step_metrics(metric_names=selected, context=context)
        metrics.update(metric_values)
        return metrics

    def default_plot_names(self) -> list[str]:
        return []

    def resolve_plot_names(self, cfg: Mapping[str, Any] | None = None) -> list[str]:
        src = dict(cfg or self.config_json)
        names = src.get("plot_names")
        if isinstance(names, list):
            return [str(v) for v in names]
        return []

    def apply_registered_plots(
        self,
        *,
        context: Mapping[str, Any],
        plot_names: list[str] | None = None,
    ) -> dict[str, Any]:
        selected = list(plot_names) if plot_names is not None else list(self.config_json.get("plot_names") or [])
        if not selected:
            return {}

        plot_outputs: dict[str, Any] = {}
        for plot_name in selected:
            if plot_name not in STEP_PLOT_REGISTRY:
                available = ", ".join(sorted(STEP_PLOT_REGISTRY))
                raise KeyError(f"Unknown plot '{plot_name}'. Available plots: [{available}]")
            kwargs = EvaluationRuntimeConfigResolver.merge_plot_kwargs(context=context, plot_name=plot_name)

            result = create_step_plot(plot_name).render(**kwargs)
            if result is not None:
                key = f"{plot_name}_path" if isinstance(result, str) else f"{plot_name}_result"
                plot_outputs[key] = result

        return plot_outputs

    def build_evaluator(self):
        return None

    def evaluate_from_loader(self, *, loader, cfg: dict[str, Any], module, evaluator) -> dict[str, Any]:
        raise NotImplementedError(f"{self.__class__.__name__} must implement evaluate_from_loader(...).")

    def _execute(self) -> EvaluationStepPayload:
        module = self.runtime_state.get("module")
        loader_factory = self.runtime_state.get("loader_factory")
        if module is None:
            raise RuntimeError(f"{self.__class__.__name__} runtime_state missing 'module'.")
        if not isinstance(loader_factory, LoaderFactory):
            raise RuntimeError(f"{self.__class__.__name__} runtime_state missing valid 'loader_factory'.")

        cfg = dict(self.config_json)
        provider, loader_params, loader = build_evaluation_loader_bundle(loader_factory=loader_factory, cfg=cfg)
        if not provider.include_targets:
            raise RuntimeError(f"{self.__class__.__name__} expects evaluation loader with targets enabled.")

        evaluator = self.build_evaluator()
        metrics = self.evaluate_from_loader(
            loader=loader,
            cfg=cfg,
            module=module,
            evaluator=evaluator,
        )

        if bool(loader_params.get("log_diagnostics", False)):
            diag = LoaderFactory.log_diagnostics(label="evaluate", loader_provider=provider)
            if diag:
                metrics["loader_diagnostics"] = diag

        return self.build_payload(metrics=metrics)

    def build_payload(self, *, metrics: dict, **kwargs) -> EvaluationStepPayload:
        payload = {"metrics": dict(metrics)}
        payload.update(dict(kwargs))
        return EvaluationStepPayload(**payload)
