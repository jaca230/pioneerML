from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from pioneerml.common.data_loader import BaseLoaderManager
from pioneerml.common.integration.pytorch.compilers import CompilerFactory
from pioneerml.common.integration.pytorch.models.architectures.factory import ArchitectureFactory
from pioneerml.common.integration.pytorch.modules.factory import ModuleFactory
from pioneerml.common.integration.pytorch.trainers import TrainerFactory

from pioneerml.common.pipeline.steps.step_types.model_runner.utils import (
    merge_nested_dicts,
)

from .......resolver import BasePayloadResolver


class FullTrainStateResolver(BasePayloadResolver):
    def resolve(
        self,
        *,
        payloads: Mapping[str, Any] | None,
        runtime_state: dict[str, Any],
    ) -> None:
        runtime_state["training_context"] = f"train_{self.step.__class__.__name__.lower()}"
        hpo_params = self._resolve_hpo_params(payloads=payloads)
        runtime_state["hpo_params"] = hpo_params
        runtime_state["upstream_payloads"] = dict(payloads or {})
        cfg = merge_nested_dicts(base=dict(self.step.config_json), override=hpo_params)
        runtime_state["resolved_training_config"] = cfg

        loader_manager = runtime_state.get("loader_manager")
        if not isinstance(loader_manager, BaseLoaderManager):
            raise RuntimeError(f"{self.step.__class__.__name__} runtime_state missing valid 'loader_manager'.")
        train_provider, train_params, train_loader = loader_manager.build_dataloader(
            purpose="train",
            default_shuffle=True,
        )
        val_provider, val_params, val_loader = loader_manager.build_dataloader(
            purpose="val",
            default_shuffle=False,
        )
        if not train_provider.include_targets or not val_provider.include_targets:
            raise RuntimeError(f"{self.step.__class__.__name__} expects train/val loaders with targets enabled.")
        runtime_state["train_provider"] = train_provider
        runtime_state["val_provider"] = val_provider
        runtime_state["train_params"] = train_params
        runtime_state["val_params"] = val_params
        runtime_state["train_loader"] = train_loader
        runtime_state["val_loader"] = val_loader

        architecture_cfg = dict(cfg.get("architecture") or {})
        module_cfg = dict(cfg.get("module") or {})
        trainer_cfg = dict(cfg.get("trainer") or {})
        architecture_name = str(architecture_cfg["type"]).strip()
        module_name = str(module_cfg["type"]).strip()
        if architecture_name == "":
            raise RuntimeError("training.architecture.type must be configured.")
        if module_name == "":
            raise RuntimeError("training.module.type must be configured.")
        model = ArchitectureFactory(architecture_name=architecture_name).build(
            config=dict(architecture_cfg.get("config") or {})
        )
        compiler_block = dict(cfg.get("compiler") or {})
        compile_cfg = dict(compiler_block.get("config") or {})
        compiler_name = str(compiler_block["type"]).strip()
        compiler = CompilerFactory(compiler_name=compiler_name).build(config=compile_cfg)
        model = compiler.compile(model=model, context=str(runtime_state["training_context"]))
        module_params = dict(module_cfg.get("config") or {})
        module_params["model"] = model
        module = ModuleFactory(module_name=module_name).build(config=module_params)
        runtime_state["module"] = module

        trainer_name = str(trainer_cfg["type"]).strip()
        trainer_inner_cfg = dict(trainer_cfg.get("config") or {})
        trainer_kwargs = dict(trainer_inner_cfg.get("trainer_kwargs") or {})
        if trainer_inner_cfg.get("max_epochs") is not None:
            trainer_kwargs.setdefault("max_epochs", int(trainer_inner_cfg["max_epochs"]))
        if trainer_inner_cfg.get("grad_clip") is not None:
            trainer_kwargs.setdefault("gradient_clip_val", float(trainer_inner_cfg["grad_clip"]))
        trainer = TrainerFactory(trainer_name=trainer_name).build(
            config={
                "trainer_kwargs": trainer_kwargs,
                "early_stopping_cfg": dict(trainer_inner_cfg.get("early_stopping") or {}),
            }
        )
        runtime_state["trainer"] = trainer

    def _resolve_hpo_params(self, *, payloads: Mapping[str, Any] | None) -> dict[str, Any]:
        if isinstance(payloads, Mapping):
            hpo_payload = payloads.get("hpo") or payloads.get("hpo_payload")
            if isinstance(hpo_payload, Mapping):
                payload_params = hpo_payload.get("hpo_params")
                if isinstance(payload_params, Mapping):
                    return dict(payload_params)
            direct = payloads.get("hpo_params")
            if isinstance(direct, Mapping):
                return dict(direct)

        hpo_payload = getattr(self.step, "hpo_payload", None)
        if isinstance(hpo_payload, Mapping):
            payload_params = hpo_payload.get("hpo_params")
            if isinstance(payload_params, Mapping):
                return dict(payload_params)
        raw = getattr(self.step, "hpo_params", None)
        if raw is None:
            return {}
        if isinstance(raw, Mapping):
            nested = raw.get("hpo_params") if "hpo_params" in raw else None
            if isinstance(nested, Mapping):
                return dict(nested)
            return dict(raw)
        raise TypeError(f"{self.step.__class__.__name__}.hpo_params must be a mapping when provided.")
