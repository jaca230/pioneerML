from __future__ import annotations

from abc import abstractmethod

import optuna

from pioneerml.common.data_loader import LoaderFactory

from ..base_training_step import BaseTrainingStep
from .payloads import HPOStepPayload
from .resolvers import HPORuntimeConfigResolver, HPORuntimeStateResolver
from .utils import (
    best_batch_size,
    build_hpo_trainer_kwargs,
    optimize_study,
    resolve_batch_size_for_trial,
    with_trial_loader_split_seed,
)
from ..utils.training_runtime_utils import fit_module_with_loaders, maybe_compile_model, merge_nested_dicts


class BaseHPOStep(BaseTrainingStep):
    config_resolver_classes = BaseTrainingStep.config_resolver_classes + (HPORuntimeConfigResolver,)
    payload_resolver_classes = BaseTrainingStep.payload_resolver_classes + (HPORuntimeStateResolver,)

    def default_config(self) -> dict:
        overrides = {
            "enabled": True,
            "n_trials": 20,
            "direction": "minimize",
            "seed": None,
            "study_name": "hpo_study",
            "storage": None,
            "fallback_dir": None,
            "allow_schema_fallback": True,
            "batch_size": {"min_exp": 5, "max_exp": 7},
            "loader_split_seed_mode": "fixed",
            "loader_split_seed": None,
            "loader_config": {
                "base": {
                    "sample_fraction": 0.25,
                }
            },
            "model": {},
            "train": {},
        }
        return merge_nested_dicts(base=super().default_config(), override=overrides)

    @abstractmethod
    def build_objective_adapter(self):
        raise NotImplementedError

    def _build_hpo_loaders(self, *, cfg: dict, batch_size: int):
        loader_factory = self.runtime_state.get("loader_factory")
        if not isinstance(loader_factory, LoaderFactory):
            raise RuntimeError(f"{self.__class__.__name__} runtime_state missing valid 'loader_factory'.")

        train_params = LoaderFactory._resolve_loader_params(cfg, purpose="train", forced_batch_size=batch_size)
        val_params = LoaderFactory._resolve_loader_params(cfg, purpose="val", forced_batch_size=batch_size)
        train_provider = loader_factory.build_loader(loader_params=train_params)
        val_provider = loader_factory.build_loader(loader_params=val_params)
        if not train_provider.include_targets or not val_provider.include_targets:
            raise RuntimeError(f"{self.__class__.__name__} expects train/val loaders with targets enabled.")

        train_loader = train_provider.make_dataloader(shuffle_batches=bool(train_params.get("shuffle_batches", True)))
        val_loader = val_provider.make_dataloader(shuffle_batches=bool(val_params.get("shuffle_batches", False)))
        return train_loader, val_loader

    def _execute(self) -> HPOStepPayload:
        hpo_enabled = bool(self.runtime_state.get("hpo_enabled", True))
        if not hpo_enabled:
            return self.build_payload(hpo_params={})

        self.apply_warning_filter()
        cfg = dict(self.config_json)
        fixed_batch_size, min_exp, max_exp = HPORuntimeConfigResolver.resolve_batch_size_search(cfg)
        trainer_kwargs = build_hpo_trainer_kwargs(cfg)
        model_search_cfg = dict(cfg.get("model") or {})
        train_search_cfg = dict(cfg.get("train") or {})
        adapter = self.runtime_state.get("objective_adapter")
        training_context = self.runtime_state.get("training_context")
        if adapter is None:
            raise RuntimeError(f"{self.__class__.__name__} runtime_state missing 'objective_adapter'.")
        if not isinstance(training_context, str) or training_context == "":
            raise RuntimeError(f"{self.__class__.__name__} runtime_state missing valid 'training_context'.")

        def objective(trial: optuna.Trial) -> float:
            batch_size = resolve_batch_size_for_trial(
                trial=trial,
                fixed_batch_size=fixed_batch_size,
                min_exp=min_exp,
                max_exp=max_exp,
            )
            trial_cfg = with_trial_loader_split_seed(cfg=cfg, trial_number=int(trial.number))
            model_params = adapter.suggest_model_params(
                trial=trial,
                model_search_cfg=model_search_cfg,
            )
            train_params = adapter.suggest_train_params(
                trial=trial,
                train_search_cfg=train_search_cfg if train_search_cfg else cfg,
            )
            model = adapter.build_model(
                model_cfg=model_params,
                compile_cfg=None,
                context=training_context,
            )
            model = maybe_compile_model(model, cfg.get("compile"), context=training_context)
            module_train_cfg = adapter.build_hpo_module_train_cfg(
                cfg=cfg,
                train_params=train_params,
            )
            module = adapter.build_module(
                model=model,
                train_cfg=module_train_cfg,
            )
            train_loader, val_loader = self._build_hpo_loaders(cfg=trial_cfg, batch_size=batch_size)
            fit_module_with_loaders(
                module=module,
                train_loader=train_loader,
                val_loader=val_loader,
                max_epochs=int(trial_cfg["max_epochs"]),
                grad_clip=float(trial_cfg["grad_clip"]) if trial_cfg.get("grad_clip") is not None else None,
                trainer_kwargs=trainer_kwargs,
                early_stopping_cfg=dict(trial_cfg.get("early_stopping") or {}),
            )
            return adapter.objective_from_module(module)

        sampler = None
        if cfg.get("seed") is not None:
            sampler = optuna.samplers.TPESampler(seed=int(cfg["seed"]))
        study, storage_used = optimize_study(
            objective=objective,
            n_trials=int(cfg["n_trials"]),
            study_name=str(cfg["study_name"]),
            direction=str(cfg["direction"]),
            storage=cfg.get("storage"),
            fallback_dir=cfg.get("fallback_dir"),
            allow_schema_fallback=bool(cfg.get("allow_schema_fallback", True)),
            sampler=sampler,
        )
        result = adapter.build_hpo_result(
            study=study,
            storage_used=storage_used,
            batch_size=best_batch_size(study=study, fixed_batch_size=fixed_batch_size, min_exp=min_exp),
            cfg=cfg,
        )
        return self.build_payload(hpo_params=result)

    @staticmethod
    def build_payload(*, hpo_params: dict, **kwargs) -> HPOStepPayload:
        payload = {"hpo_params": dict(hpo_params)}
        payload.update(dict(kwargs))
        return HPOStepPayload(**payload)
