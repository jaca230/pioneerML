from __future__ import annotations

from pioneerml.common.evaluation.evaluators import BaseEvaluator, EvaluatorFactory
from .payloads import EvaluationStepPayload
from .resolvers import EvaluationConfigResolver, EvaluationStateResolver
from ..utils import log_loader_diagnostics, merge_nested_dicts

from ..base_model_runner_step import BaseModelRunnerStep


class BaseEvaluationStep(BaseModelRunnerStep):
    DEFAULT_CONFIG = merge_nested_dicts(
        base=BaseModelRunnerStep.DEFAULT_CONFIG,
        override={
            "enabled": True,
            "evaluator": {"type": "required", "config": {}},
            "loader_manager": {
                "config": {
                    "defaults": {"type": "group_classifier", "config": {"batch_size": 1}},
                    "loaders": {
                        "test_loader": {
                            "config": {
                                "mode": "train",
                                "split": "test",
                                "shuffle_batches": False,
                                "log_diagnostics": False,
                            },
                        },
                    },
                },
            },
        },
    )
    config_resolver_classes = BaseModelRunnerStep.config_resolver_classes + (EvaluationConfigResolver,)
    payload_resolver_classes = BaseModelRunnerStep.payload_resolver_classes + (EvaluationStateResolver,)

    def _execute(self) -> EvaluationStepPayload:
        if bool(self.runtime_state.get("evaluation_disabled", False)):
            return self.build_payload(metrics={"skipped": True})

        module = self.runtime_state.get("module")
        provider = self.runtime_state.get("evaluation_provider")
        loader_params = self.runtime_state.get("evaluation_loader_params")
        loader = self.runtime_state.get("evaluation_loader")
        if module is None:
            raise RuntimeError(f"{self.__class__.__name__} runtime_state missing 'module'.")
        if provider is None:
            raise RuntimeError(f"{self.__class__.__name__} runtime_state missing 'evaluation_provider'.")
        if not isinstance(loader_params, dict):
            raise RuntimeError(f"{self.__class__.__name__} runtime_state missing 'evaluation_loader_params'.")
        if loader is None:
            raise RuntimeError(f"{self.__class__.__name__} runtime_state missing 'evaluation_loader'.")

        cfg = dict(self.config_json)
        evaluator_cfg = dict(cfg.get("evaluator") or {})
        evaluator_type = str(evaluator_cfg["type"]).strip()
        evaluator_config = dict(evaluator_cfg.get("config") or {})
        evaluator_factory = EvaluatorFactory(evaluator_name=evaluator_type)
        evaluator = evaluator_factory.build(config=evaluator_config)
        if not isinstance(evaluator, BaseEvaluator):
            raise RuntimeError(f"{self.__class__.__name__} evaluator_factory must build BaseEvaluator.")
        evaluator_run_cfg = dict(cfg)
        evaluator_run_cfg.update(evaluator_config)
        metrics = evaluator.evaluate(module=module, loader=loader, config=evaluator_run_cfg)
        if not isinstance(metrics, dict):
            raise RuntimeError(f"{self.__class__.__name__} evaluator must return dict metrics.")

        if bool(loader_params.get("log_diagnostics", False)):
            diag = log_loader_diagnostics(label="evaluate", loader_provider=provider)
            if diag:
                metrics["loader_diagnostics"] = diag

        return self.build_payload(metrics=metrics)

    def build_payload(self, *, metrics: dict, **kwargs) -> EvaluationStepPayload:
        payload = {"metrics": dict(metrics)}
        payload.update(dict(kwargs))
        return EvaluationStepPayload(**payload)
