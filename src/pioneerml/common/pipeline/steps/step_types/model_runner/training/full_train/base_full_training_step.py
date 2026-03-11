from __future__ import annotations

from pioneerml.common.data_loader import LoaderFactory

from ..base_training_step import BaseTrainingStep
from .payloads import TrainingStepPayload
from .resolvers.payload import FullTrainRuntimeStateResolver
from ..utils.training_runtime_utils import (
    build_train_val_providers,
    build_training_module,
    fit_module_with_loaders,
    resolve_effective_training_config,
    validate_train_val_providers,
)


class BaseFullTrainingStep(BaseTrainingStep):
    payload_resolver_classes = BaseTrainingStep.payload_resolver_classes + (FullTrainRuntimeStateResolver,)

    def build_payload(
        self,
        *,
        module,
        training_context: str,
        hpo_params: dict | None = None,
        upstream_payloads: dict | None = None,
    ) -> TrainingStepPayload:
        return TrainingStepPayload(
            module=module,
            training_context=str(training_context),
            hpo_params=dict(hpo_params or {}),
            upstream_payloads=dict(upstream_payloads or {}),
        )

    def _execute(self):
        self.apply_warning_filter()
        cfg = resolve_effective_training_config(
            config=self.config_json,
            runtime_overrides=self.runtime_state.get("hpo_params"),
        )
        objective_adapter = self.runtime_state.get("objective_adapter")
        loader_factory = self.runtime_state.get("loader_factory")
        training_context = self.runtime_state.get("training_context")
        hpo_params = self.runtime_state.get("hpo_params")
        upstream_payloads = self.runtime_state.get("upstream_payloads")
        if objective_adapter is None:
            raise RuntimeError(f"{self.__class__.__name__} runtime_state missing 'objective_adapter'.")
        if not isinstance(loader_factory, LoaderFactory):
            raise RuntimeError(f"{self.__class__.__name__} runtime_state missing valid 'loader_factory'.")
        if not isinstance(training_context, str) or training_context == "":
            raise RuntimeError(f"{self.__class__.__name__} runtime_state missing valid 'training_context'.")
        if hpo_params is not None and not isinstance(hpo_params, dict):
            raise RuntimeError(f"{self.__class__.__name__} runtime_state has invalid 'hpo_params'.")
        if upstream_payloads is not None and not isinstance(upstream_payloads, dict):
            raise RuntimeError(f"{self.__class__.__name__} runtime_state has invalid 'upstream_payloads'.")

        module = build_training_module(
            objective_adapter=objective_adapter,
            cfg=cfg,
            context=training_context,
        )
        train_provider, val_provider, train_params, val_params = build_train_val_providers(
            loader_factory=loader_factory,
            cfg=cfg,
        )
        validate_train_val_providers(step=self, train_provider=train_provider, val_provider=val_provider)
        train_loader = train_provider.make_dataloader(shuffle_batches=bool(train_params.get("shuffle_batches", True)))
        val_loader = val_provider.make_dataloader(shuffle_batches=bool(val_params.get("shuffle_batches", False)))
        module = fit_module_with_loaders(
            module=module,
            train_loader=train_loader,
            val_loader=val_loader,
            max_epochs=int(cfg["max_epochs"]),
            grad_clip=float(cfg["grad_clip"]) if cfg.get("grad_clip") is not None else None,
            trainer_kwargs=dict(cfg.get("trainer_kwargs") or {}),
            early_stopping_cfg=dict(cfg.get("early_stopping") or {}),
        )
        if bool(train_params.get("log_diagnostics", False)):
            LoaderFactory.log_diagnostics(label="train", loader_provider=train_provider)
        if bool(val_params.get("log_diagnostics", False)):
            LoaderFactory.log_diagnostics(label="val", loader_provider=val_provider)
        return self.build_payload(
            module=module,
            training_context=training_context,
            hpo_params=hpo_params,
            upstream_payloads=upstream_payloads,
        )
