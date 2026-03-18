from __future__ import annotations

from pioneerml.common.evaluation.evaluators import BaseEvaluator, EvaluatorFactory
from .payloads import EvaluationStepPayload
from .resolvers import EvaluationRuntimeConfigResolver, EvaluationRuntimeStateResolver
from ..utils import build_loader_bundle, merge_nested_dicts

from ..base_model_runner_step import BaseModelRunnerStep
from pioneerml.common.data_loader import LoaderFactory


class BaseEvaluationStep(BaseModelRunnerStep):
    DEFAULT_CONFIG = merge_nested_dicts(
        base=BaseModelRunnerStep.DEFAULT_CONFIG,
        override={
            "evaluator_name": None,
            "evaluator_config": {},
            "metrics": [],
            "plots": [],
            "loader_config": {
                "base": {"batch_size": 1},
                "test": {"mode": "train", "split": "test", "shuffle_batches": False, "log_diagnostics": False},
            },
        },
    )
    config_resolver_classes = BaseModelRunnerStep.config_resolver_classes + (EvaluationRuntimeConfigResolver,)
    payload_resolver_classes = BaseModelRunnerStep.payload_resolver_classes + (EvaluationRuntimeStateResolver,)

    def _execute(self) -> EvaluationStepPayload:
        module = self.runtime_state.get("module")
        loader_factory = self.runtime_state.get("loader_factory")
        if module is None:
            raise RuntimeError(f"{self.__class__.__name__} runtime_state missing 'module'.")
        if not isinstance(loader_factory, LoaderFactory):
            raise RuntimeError(f"{self.__class__.__name__} runtime_state missing valid 'loader_factory'.")

        cfg = dict(self.config_json)
        provider, loader_params, loader = build_loader_bundle(
            loader_factory=loader_factory,
            cfg=cfg,
            purpose="test",
            default_shuffle=False,
        )
        if not provider.include_targets:
            raise RuntimeError(f"{self.__class__.__name__} expects evaluation loader with targets enabled.")

        evaluator_factory = EvaluatorFactory(evaluator_name=str(cfg["evaluator_name"]))
        evaluator = evaluator_factory.build_evaluator(evaluator_params=dict(cfg.get("evaluator_config") or {}))
        if not isinstance(evaluator, BaseEvaluator):
            raise RuntimeError(f"{self.__class__.__name__} evaluator_factory must build BaseEvaluator.")
        evaluator_cfg = dict(cfg)
        evaluator_cfg.update(dict(cfg.get("evaluator_config") or {}))
        metrics = evaluator.evaluate(module=module, loader=loader, config=evaluator_cfg)
        if not isinstance(metrics, dict):
            raise RuntimeError(f"{self.__class__.__name__} evaluator must return dict metrics.")

        if bool(loader_params.get("log_diagnostics", False)):
            diag = LoaderFactory.log_diagnostics(label="evaluate", loader_provider=provider)
            if diag:
                metrics["loader_diagnostics"] = diag

        return self.build_payload(metrics=metrics)

    def build_payload(self, *, metrics: dict, **kwargs) -> EvaluationStepPayload:
        payload = {"metrics": dict(metrics)}
        payload.update(dict(kwargs))
        return EvaluationStepPayload(**payload)
