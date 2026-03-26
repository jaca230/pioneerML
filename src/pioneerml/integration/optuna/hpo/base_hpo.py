from __future__ import annotations

from abc import ABC
from collections.abc import Mapping
from typing import Any

import optuna

from .objective import BaseObjective
from .search_space import BaseSearchSpace


class BaseHPO(ABC):
    def __init__(
        self,
        *,
        objective: BaseObjective,
        search_space: BaseSearchSpace,
        search_space_config: Mapping[str, Any] | None = None,
        enabled: bool = True,
        n_trials: int = 20,
        direction: str = "minimize",
        seed: int | None = None,
        study_name: str = "hpo_study",
        storage: str | None = None,
        fallback_dir: str | None = None,
        allow_schema_fallback: bool = True,
        loader_split_seed_mode: str = "fixed",
        loader_split_seed: int | None = None,
        max_train_batches: int | None = None,
        max_val_batches: int | None = None,
    ) -> None:
        self.objective = objective
        self.search_space = search_space
        self.search_space_config = dict(search_space_config or {})
        self.enabled = bool(enabled)
        self.n_trials = int(n_trials)
        self.direction = str(direction)
        self.seed = (None if seed is None else int(seed))
        self.study_name = str(study_name)
        self.storage = (None if storage is None else str(storage))
        self.fallback_dir = (None if fallback_dir is None else str(fallback_dir))
        self.allow_schema_fallback = bool(allow_schema_fallback)
        self.loader_split_seed_mode = str(loader_split_seed_mode).strip().lower()
        self.loader_split_seed = (None if loader_split_seed is None else int(loader_split_seed))
        self.max_train_batches = (None if max_train_batches is None else int(max_train_batches))
        self.max_val_batches = (None if max_val_batches is None else int(max_val_batches))

    def suggest(self, *, trial: optuna.Trial) -> dict[str, Any]:
        return self.search_space.suggest(trial=trial, search_space=self.search_space_config)

    def partition_suggested_params(
        self,
        *,
        suggested: Mapping[str, Any],
        model_cls: Any,
        module_cls: Any,
    ) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
        return self.search_space.partition_suggested_params(
            suggested=suggested,
            model_cls=model_cls,
            module_cls=module_cls,
        )

    def objective_from_module(self, module: Any) -> float:
        return float(self.objective.objective_from_module(module))

    def runtime_config(self) -> dict[str, Any]:
        return {
            "enabled": self.enabled,
            "n_trials": int(self.n_trials),
            "direction": str(self.direction),
            "seed": self.seed,
            "study_name": str(self.study_name),
            "storage": self.storage,
            "fallback_dir": self.fallback_dir,
            "allow_schema_fallback": bool(self.allow_schema_fallback),
            "loader_split_seed_mode": str(self.loader_split_seed_mode),
            "loader_split_seed": self.loader_split_seed,
            "max_train_batches": self.max_train_batches,
            "max_val_batches": self.max_val_batches,
        }
