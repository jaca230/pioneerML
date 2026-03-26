from __future__ import annotations

import optuna

from pioneerml.integration.optuna.hpo import BaseHPO
from pioneerml.integration.pytorch.compilers import CompilerFactory
from pioneerml.integration.pytorch.models.architectures.factory import ArchitectureFactory
from pioneerml.integration.pytorch.modules.factory import ModuleFactory
from pioneerml.integration.pytorch.trainers import TrainerFactory
from pioneerml.plugin import resolve_plugin

from ..base_training_step import BaseTrainingStep
from .payloads import HPOStepPayload
from .resolvers import HPOConfigResolver, HPOStateResolver
from ...utils import merge_nested_dicts


class BaseHPOStep(BaseTrainingStep):
    config_resolver_classes = BaseTrainingStep.config_resolver_classes + (HPOConfigResolver,)
    payload_resolver_classes = BaseTrainingStep.payload_resolver_classes + (HPOStateResolver,)

    def default_config(self) -> dict:
        overrides = {
            "loader_manager": {
                "config": {
                    "defaults": {
                        "type": "group_classifier",
                        "config": {
                            "sample_fraction": 0.25,
                        },
                    },
                },
            },
            "hpo": {
                "type": "config",
                "config": {
                    "enabled": True,
                    "n_trials": 20,
                    "direction": "minimize",
                    "seed": None,
                    "study_name": "hpo_study",
                    "storage": None,
                    "fallback_dir": None,
                    "allow_schema_fallback": True,
                    "loader_split_seed_mode": "fixed",
                    "loader_split_seed": None,
                    "objective": {"type": "val_epoch", "config": {}},
                    "search_space": {
                        "type": "config",
                        "config": {
                            "search_space": {"batch_size": {"type": "exponent_int", "base": 2, "min_exp": 5, "max_exp": 7}}
                        },
                    },
                },
            },
        }
        return merge_nested_dicts(base=super().default_config(), override=overrides)

    def _execute(self) -> HPOStepPayload:
        hpo_enabled = bool(self.runtime_state.get("hpo_enabled", True))
        if not hpo_enabled:
            return self.build_payload(hpo_params={})

        self.apply_warning_filter()
        cfg = dict(self.config_json)
        hpo_cfg = dict(dict(cfg.get("hpo") or {}).get("config") or {})
        architecture_cfg = dict(cfg.get("architecture") or {})
        module_cfg = dict(cfg.get("module") or {})
        trainer_cfg = dict(cfg.get("trainer") or {})
        compiler_block = dict(cfg.get("compiler") or {})
        compile_cfg = dict(compiler_block.get("config") or {})
        architecture_name = str(architecture_cfg["type"]).strip()
        module_name = str(module_cfg["type"]).strip()
        if architecture_name == "":
            raise RuntimeError("training.hpo.architecture.type must be configured.")
        if module_name == "":
            raise RuntimeError("training.hpo.module.type must be configured.")

        architecture_factory = ArchitectureFactory(architecture_name=architecture_name)
        module_factory = ModuleFactory(module_name=module_name)

        hpo_plugin = self.runtime_state.get("hpo")
        hpo_loader_builder = self.runtime_state.get("hpo_loader_builder")
        training_context = self.runtime_state.get("training_context")
        batch_size_for_trial = self.runtime_state.get("hpo_batch_size_for_trial")
        best_batch_size = self.runtime_state.get("hpo_best_batch_size")
        trial_cfg_builder = self.runtime_state.get("hpo_trial_cfg_builder")
        trainer_kwargs_builder = self.runtime_state.get("hpo_trainer_kwargs_builder")
        study_optimizer = self.runtime_state.get("hpo_study_optimizer")
        result_builder = self.runtime_state.get("hpo_result_builder")

        if not isinstance(hpo_plugin, BaseHPO):
            raise RuntimeError(f"{self.__class__.__name__} runtime_state missing 'hpo'.")
        if not callable(hpo_loader_builder):
            raise RuntimeError(f"{self.__class__.__name__} runtime_state missing callable 'hpo_loader_builder'.")
        if not isinstance(training_context, str) or training_context == "":
            raise RuntimeError(f"{self.__class__.__name__} runtime_state missing valid 'training_context'.")
        if not callable(batch_size_for_trial):
            raise RuntimeError(f"{self.__class__.__name__} runtime_state missing callable 'hpo_batch_size_for_trial'.")
        if not callable(best_batch_size):
            raise RuntimeError(f"{self.__class__.__name__} runtime_state missing callable 'hpo_best_batch_size'.")
        if not callable(trial_cfg_builder):
            raise RuntimeError(f"{self.__class__.__name__} runtime_state missing callable 'hpo_trial_cfg_builder'.")
        if not callable(trainer_kwargs_builder):
            raise RuntimeError(
                f"{self.__class__.__name__} runtime_state missing callable 'hpo_trainer_kwargs_builder'."
            )
        if not callable(study_optimizer):
            raise RuntimeError(f"{self.__class__.__name__} runtime_state missing callable 'hpo_study_optimizer'.")
        if not callable(result_builder):
            raise RuntimeError(f"{self.__class__.__name__} runtime_state missing callable 'hpo_result_builder'.")

        def objective(trial: optuna.Trial) -> float:
            suggested = hpo_plugin.suggest(trial=trial)
            architecture_cls = resolve_plugin(namespace="architecture", name=architecture_name)
            module_cls = resolve_plugin(namespace="module", name=module_name)
            model_updates, module_updates, runtime_updates = hpo_plugin.partition_suggested_params(
                suggested=suggested,
                model_cls=architecture_cls,
                module_cls=module_cls,
            )
            trial_batch_size = runtime_updates.pop("batch_size", None)
            if trial_batch_size is None:
                batch_size = int(batch_size_for_trial(trial))
            else:
                batch_size = max(1, int(trial_batch_size))
            trial_cfg = trial_cfg_builder(int(trial.number))
            model_params = merge_nested_dicts(
                base=dict(architecture_cfg.get("config") or {}),
                override=model_updates,
            )
            model = architecture_factory.build(config=model_params)
            compiler_name = str(compiler_block["type"]).strip()
            compiler = CompilerFactory(compiler_name=compiler_name).build(config=compile_cfg)
            model = compiler.compile(model=model, context=training_context)

            module_params = merge_nested_dicts(
                base=dict(module_cfg.get("config") or {}),
                override=module_updates,
            )
            module_params["model"] = model
            module = module_factory.build(config=module_params)

            train_loader, val_loader = hpo_loader_builder(cfg=trial_cfg, batch_size=batch_size)
            trial_module_cfg = merge_nested_dicts(
                base=dict(trial_cfg.get("trainer") or {}),
                override=runtime_updates,
            )
            trainer_kwargs = trainer_kwargs_builder(trial_cfg)
            trial_trainer_cfg = dict(trial_module_cfg.get("config") or {})
            trial_trainer_cfg["trainer_kwargs"] = merge_nested_dicts(
                base=dict(trial_trainer_cfg.get("trainer_kwargs") or {}),
                override=trainer_kwargs,
            )
            trial_trainer_kwargs = dict(trial_trainer_cfg.get("trainer_kwargs") or {})
            if trial_trainer_cfg.get("max_epochs") is not None:
                trial_trainer_kwargs.setdefault("max_epochs", int(trial_trainer_cfg["max_epochs"]))
            if trial_trainer_cfg.get("grad_clip") is not None:
                trial_trainer_kwargs.setdefault("gradient_clip_val", float(trial_trainer_cfg["grad_clip"]))
            trainer_name = str(trial_module_cfg.get("type", trainer_cfg["type"])).strip()
            if trainer_name == "":
                raise RuntimeError("training.hpo.trainer.type must be configured.")
            trainer = TrainerFactory(trainer_name=trainer_name).build(
                config={
                    "trainer_kwargs": trial_trainer_kwargs,
                    "early_stopping_cfg": dict(trial_trainer_cfg.get("early_stopping") or {}),
                }
            )
            trainer.fit(
                model=module,
                train_dataloaders=train_loader,
                val_dataloaders=val_loader,
            )
            return hpo_plugin.objective_from_module(module)

        sampler = None
        if hpo_cfg.get("seed") is not None:
            sampler = optuna.samplers.TPESampler(seed=int(hpo_cfg["seed"]))
        study, storage_used = study_optimizer(
            objective=objective,
            n_trials=int(hpo_cfg["n_trials"]),
            study_name=str(hpo_cfg["study_name"]),
            direction=str(hpo_cfg["direction"]),
            storage=hpo_cfg.get("storage"),
            fallback_dir=hpo_cfg.get("fallback_dir"),
            allow_schema_fallback=bool(hpo_cfg.get("allow_schema_fallback", True)),
            sampler=sampler,
        )
        result = result_builder(
            study=study,
            storage_used=storage_used,
            batch_size=(
                int(study.best_params["batch_size"])
                if "batch_size" in dict(getattr(study, "best_params", {}) or {})
                else int(best_batch_size(study))
            ),
        )
        return self.build_payload(hpo_params=result)

    @staticmethod
    def build_payload(*, hpo_params: dict, **kwargs) -> HPOStepPayload:
        payload = {"hpo_params": dict(hpo_params)}
        payload.update(dict(kwargs))
        return HPOStepPayload(**payload)
