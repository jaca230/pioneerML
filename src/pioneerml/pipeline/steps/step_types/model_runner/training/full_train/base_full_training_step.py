from __future__ import annotations

from pioneerml.pipeline.steps.step_types.model_runner.utils import log_loader_diagnostics

from ..base_training_step import BaseTrainingStep
from .payloads import TrainingStepPayload
from .resolvers.payload import FullTrainStateResolver


class BaseFullTrainingStep(BaseTrainingStep):
    payload_resolver_classes = BaseTrainingStep.payload_resolver_classes + (FullTrainStateResolver,)

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
        module = self.runtime_state.get("module")
        trainer = self.runtime_state.get("trainer")
        train_loader = self.runtime_state.get("train_loader")
        val_loader = self.runtime_state.get("val_loader")
        train_provider = self.runtime_state.get("train_provider")
        val_provider = self.runtime_state.get("val_provider")
        train_params = self.runtime_state.get("train_params")
        val_params = self.runtime_state.get("val_params")
        training_context = self.runtime_state.get("training_context")
        hpo_params = self.runtime_state.get("hpo_params")
        upstream_payloads = self.runtime_state.get("upstream_payloads")
        if module is None:
            raise RuntimeError(f"{self.__class__.__name__} runtime_state missing 'module'.")
        if trainer is None:
            raise RuntimeError(f"{self.__class__.__name__} runtime_state missing 'trainer'.")
        if train_loader is None or val_loader is None:
            raise RuntimeError(f"{self.__class__.__name__} runtime_state missing train/val loaders.")
        if train_provider is None or val_provider is None:
            raise RuntimeError(f"{self.__class__.__name__} runtime_state missing train/val providers.")
        if not isinstance(train_params, dict) or not isinstance(val_params, dict):
            raise RuntimeError(f"{self.__class__.__name__} runtime_state missing train/val params.")
        if not isinstance(training_context, str) or training_context == "":
            raise RuntimeError(f"{self.__class__.__name__} runtime_state missing valid 'training_context'.")
        if hpo_params is not None and not isinstance(hpo_params, dict):
            raise RuntimeError(f"{self.__class__.__name__} runtime_state has invalid 'hpo_params'.")
        if upstream_payloads is not None and not isinstance(upstream_payloads, dict):
            raise RuntimeError(f"{self.__class__.__name__} runtime_state has invalid 'upstream_payloads'.")

        trainer.fit(
            model=module,
            train_dataloaders=train_loader,
            val_dataloaders=val_loader,
        )
        if bool(train_params.get("log_diagnostics", False)):
            log_loader_diagnostics(label="train", loader_provider=train_provider)
        if bool(val_params.get("log_diagnostics", False)):
            log_loader_diagnostics(label="val", loader_provider=val_provider)
        return self.build_payload(
            module=module,
            training_context=training_context,
            hpo_params=hpo_params,
            upstream_payloads=upstream_payloads,
        )
