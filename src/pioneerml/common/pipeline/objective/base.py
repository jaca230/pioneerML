from __future__ import annotations

from abc import ABC, abstractmethod

import optuna

from pioneerml.common.pipeline.services.training.hpo.utils import suggest_range


class BaseObjectiveAdapter(ABC):
    def default_hpo_model_space(self) -> dict:
        return {}

    def default_hpo_train_space(self) -> dict:
        return {
            "lr": {"low": 1e-4, "high": 1e-2, "log": True},
            "weight_decay": {"low": 1e-6, "high": 1e-3, "log": True},
        }

    def suggest_model_params(self, *, trial: optuna.Trial, model_search_cfg: dict | None = None) -> dict:
        _ = trial
        cfg = dict(self.default_hpo_model_space())
        if model_search_cfg:
            cfg.update(dict(model_search_cfg))
        return cfg

    def suggest_train_params(self, *, trial: optuna.Trial, train_search_cfg: dict | None = None) -> dict:
        cfg = dict(self.default_hpo_train_space())
        if train_search_cfg:
            cfg.update(dict(train_search_cfg))
        lr_low, lr_high, lr_log = suggest_range(cfg, "lr", default_low=1e-4, default_high=1e-2)
        wd_low, wd_high, wd_log = suggest_range(cfg, "weight_decay", default_low=1e-6, default_high=1e-3)
        return {
            "lr": float(trial.suggest_float("lr", lr_low, lr_high, log=lr_log)),
            "weight_decay": float(trial.suggest_float("weight_decay", wd_low, wd_high, log=wd_log)),
        }

    def build_hpo_module_train_cfg(self, *, cfg: dict, train_params: dict) -> dict:
        out = {
            "threshold": float(cfg.get("threshold", 0.5)),
            "scheduler_step_size": cfg.get("scheduler_step_size"),
            "scheduler_gamma": float(cfg.get("scheduler_gamma", 0.5)),
        }
        out.update(dict(train_params or {}))
        return out

    def build_hpo_result(
        self,
        *,
        study,
        storage_used: str | None,
        batch_size: int,
        cfg: dict,
    ) -> dict:
        _ = cfg
        return {
            "study_name": study.study_name,
            "storage": storage_used,
            "batch_size": int(batch_size),
            "best_params": dict(study.best_params),
        }

    @abstractmethod
    def build_model(self, *, model_cfg: dict, compile_cfg: dict | None, context: str):
        raise NotImplementedError

    @abstractmethod
    def build_module(self, *, model, train_cfg: dict):
        raise NotImplementedError

    @abstractmethod
    def objective_from_module(self, module) -> float:
        raise NotImplementedError
