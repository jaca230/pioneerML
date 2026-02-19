from __future__ import annotations

from abc import abstractmethod

import optuna

from pioneerml.common.optuna.manager import OptunaStudyManager

from ..base_training_service import BaseTrainingService
from .utils import build_hpo_trainer_kwargs, resolve_batch_size_search


class BaseHPOService(BaseTrainingService):
    @abstractmethod
    def build_hpo_loaders(self, *, cfg: dict, batch_size: int):
        raise NotImplementedError

    def _objective_adapter(self):
        adapter = getattr(self, "objective_adapter", None)
        if adapter is None:
            raise RuntimeError(f"{self.__class__.__name__} is missing objective_adapter.")
        return adapter

    def hpo_enabled(self) -> bool:
        step_cfg = self.step_config()
        if step_cfg is None:
            return False
        return bool(step_cfg.get("enabled", True))

    def objective_context(self) -> str:
        return f"tune_{self.__class__.__name__.lower()}"

    def study_name_default(self) -> str:
        return "hpo_study"

    def resolve_batch_size_for_trial(
        self,
        *,
        trial: optuna.Trial,
        fixed_batch_size: int | None,
        min_exp: int,
        max_exp: int,
    ) -> int:
        if fixed_batch_size is not None:
            return int(fixed_batch_size)
        return int(1 << int(trial.suggest_int("batch_size_exp", min_exp, max_exp)))

    def best_batch_size(self, *, study, fixed_batch_size: int | None, min_exp: int) -> int:
        if fixed_batch_size is not None:
            return int(fixed_batch_size)
        return int(1 << int(study.best_params.get("batch_size_exp", min_exp)))

    def optimize_study(
        self,
        *,
        objective,
        n_trials: int,
        study_name: str,
        direction: str,
        storage,
        fallback_dir,
        allow_schema_fallback: bool,
        sampler,
    ):
        manager = OptunaStudyManager(
            study_name=str(study_name),
            direction=str(direction),
            storage=storage,
            fallback_dir=fallback_dir,
            allow_schema_fallback=bool(allow_schema_fallback),
        )
        study, storage_used = manager.create_or_load(sampler=sampler)
        study.optimize(objective, n_trials=int(n_trials))
        return study, storage_used

    def execute(self) -> dict:
        if not self.hpo_enabled():
            return {}

        self.apply_warning_filter()
        cfg = self.get_config()
        fixed_batch_size, min_exp, max_exp = resolve_batch_size_search(cfg)
        trainer_kwargs = build_hpo_trainer_kwargs(cfg)
        model_search_cfg = dict(cfg.get("model") or {})
        train_search_cfg = dict(cfg.get("train") or {})
        adapter = self._objective_adapter()

        def objective(trial: optuna.Trial) -> float:
            batch_size = self.resolve_batch_size_for_trial(
                trial=trial,
                fixed_batch_size=fixed_batch_size,
                min_exp=min_exp,
                max_exp=max_exp,
            )
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
                context=self.objective_context(),
            )
            model = self.compile_model(model, compile_cfg=cfg.get("compile"), context=self.objective_context())
            module_train_cfg = adapter.build_hpo_module_train_cfg(
                cfg=cfg,
                train_params=train_params,
            )
            module = adapter.build_module(
                model=model,
                train_cfg=module_train_cfg,
            )
            train_loader, val_loader = self.build_hpo_loaders(cfg=cfg, batch_size=batch_size)
            self.fit_module(
                module=module,
                train_loader=train_loader,
                val_loader=val_loader,
                max_epochs=int(cfg["max_epochs"]),
                grad_clip=float(cfg["grad_clip"]) if cfg.get("grad_clip") is not None else None,
                trainer_kwargs=trainer_kwargs,
                early_stopping_cfg=dict(cfg.get("early_stopping") or {}),
            )
            return adapter.objective_from_module(module)

        sampler = None
        if cfg.get("seed") is not None:
            sampler = optuna.samplers.TPESampler(seed=int(cfg["seed"]))
        study, storage_used = self.optimize_study(
            objective=objective,
            n_trials=int(cfg["n_trials"]),
            study_name=str(cfg.get("study_name", self.study_name_default())),
            direction=str(cfg["direction"]),
            storage=cfg.get("storage"),
            fallback_dir=cfg.get("fallback_dir"),
            allow_schema_fallback=bool(cfg.get("allow_schema_fallback", True)),
            sampler=sampler,
        )
        return adapter.build_hpo_result(
            study=study,
            storage_used=storage_used,
            batch_size=self.best_batch_size(study=study, fixed_batch_size=fixed_batch_size, min_exp=min_exp),
            cfg=cfg,
        )
