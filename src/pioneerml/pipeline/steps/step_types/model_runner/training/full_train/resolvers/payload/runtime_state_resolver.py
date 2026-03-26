from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from pioneerml.data_loader import BaseLoaderManager, LoaderManagerFactory
from pioneerml.integration.optuna.hpo.search_space import BaseSearchSpace
from pioneerml.integration.pytorch.compilers import CompilerFactory
from pioneerml.integration.pytorch.models.architectures.factory import ArchitectureFactory
from pioneerml.integration.pytorch.modules.factory import ModuleFactory
from pioneerml.integration.pytorch.trainers import TrainerFactory
from pioneerml.plugin import resolve_plugin

from pioneerml.pipeline.steps.step_types.model_runner.utils import (
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
        hpo_overrides = self._resolve_hpo_overrides(
            base_cfg=dict(self.step.config_json),
            hpo_params=hpo_params,
        )
        runtime_state["hpo_params"] = hpo_params
        runtime_state["hpo_overrides"] = hpo_overrides
        runtime_state["upstream_payloads"] = dict(payloads or {})
        cfg = merge_nested_dicts(base=dict(self.step.config_json), override=hpo_overrides)
        runtime_state["resolved_training_config"] = cfg

        loader_manager = runtime_state.get("loader_manager")
        if not isinstance(loader_manager, BaseLoaderManager):
            raise RuntimeError(f"{self.step.__class__.__name__} runtime_state missing valid 'loader_manager'.")
        loader_manager_block = dict(cfg.get("loader_manager") or {})
        manager_type = str(loader_manager_block.get("type") or "").strip()
        if manager_type == "":
            manager_type = str(getattr(loader_manager, "plugin_name", "") or "").strip()
        if manager_type == "":
            raise RuntimeError(
                f"{self.step.__class__.__name__} requires non-empty 'loader_manager.type' in resolved config."
            )
        merged_loader_manager_cfg = merge_nested_dicts(
            base=dict(loader_manager.config),
            override=dict(loader_manager_block.get("config") or {}),
        )
        resolved_loader_manager = LoaderManagerFactory(loader_manager_name=manager_type).build(
            config=merged_loader_manager_cfg
        )
        if not isinstance(resolved_loader_manager, BaseLoaderManager):
            raise RuntimeError(
                f"{self.step.__class__.__name__} resolved loader_manager must be BaseLoaderManager."
            )
        runtime_state["loader_manager"] = resolved_loader_manager
        runtime_state["loader_factory"] = resolved_loader_manager.loader_factory

        train_provider, train_params, train_loader = resolved_loader_manager.build_dataloader(
            purpose="train",
            default_shuffle=True,
        )
        val_provider, val_params, val_loader = resolved_loader_manager.build_dataloader(
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

    def _resolve_hpo_overrides(
        self,
        *,
        base_cfg: dict[str, Any],
        hpo_params: Mapping[str, Any] | None,
    ) -> dict[str, Any]:
        if not isinstance(hpo_params, Mapping) or not hpo_params:
            return {}

        suggested = hpo_params.get("best_params")
        if isinstance(suggested, Mapping):
            suggested_params = dict(suggested)
        else:
            suggested_params = dict(hpo_params)
        if not suggested_params:
            return {}

        architecture_name = str(dict(base_cfg.get("architecture") or {}).get("type") or "").strip()
        module_name = str(dict(base_cfg.get("module") or {}).get("type") or "").strip()
        if architecture_name == "" or module_name == "":
            return {}

        architecture_cls = resolve_plugin(namespace="architecture", name=architecture_name)
        module_cls = resolve_plugin(namespace="module", name=module_name)
        model_updates, module_updates, runtime_updates = BaseSearchSpace.partition_suggested_params(
            suggested=suggested_params,
            model_cls=architecture_cls,
            module_cls=module_cls,
        )

        overrides: dict[str, Any] = {}
        if model_updates:
            overrides["architecture"] = {"config": dict(model_updates)}
        if module_updates:
            overrides["module"] = {"config": dict(module_updates)}

        batch_size = runtime_updates.get("batch_size")
        if batch_size is not None:
            overrides = merge_nested_dicts(
                base=overrides,
                override={
                    "loader_manager": {
                        "config": {
                            "defaults": {
                                "config": {
                                    "batch_size": max(1, int(batch_size)),
                                }
                            }
                        }
                    }
                },
            )

        return overrides
