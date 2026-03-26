from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from .base_hpo import BaseHPO
from .factory.registry import REGISTRY as HPO_REGISTRY
from .objective import BaseObjective, ObjectiveFactory
from .search_space import BaseSearchSpace, SearchSpaceFactory


@HPO_REGISTRY.register("config")
class ConfigHPO(BaseHPO):
    def __init__(
        self,
        *,
        objective: Mapping[str, Any],
        search_space: Mapping[str, Any],
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
        objective_block = dict(objective or {})
        search_space_block = dict(search_space or {})

        objective_type = str(objective_block.get("type") or "").strip()
        if objective_type == "":
            raise ValueError("hpo.config.objective.type must be a non-empty string.")
        objective_cfg = objective_block.get("config", {})
        if not isinstance(objective_cfg, Mapping):
            raise TypeError("hpo.config.objective.config must be a mapping.")
        objective_plugin = ObjectiveFactory(objective_name=objective_type).build(config=dict(objective_cfg))
        if not isinstance(objective_plugin, BaseObjective):
            raise TypeError(
                f"Objective plugin '{objective_type}' must build BaseObjective, got {type(objective_plugin).__name__}."
            )

        search_space_type = str(search_space_block.get("type") or "").strip()
        if search_space_type == "":
            raise ValueError("hpo.config.search_space.type must be a non-empty string.")
        search_space_cfg = search_space_block.get("config", {})
        if not isinstance(search_space_cfg, Mapping):
            raise TypeError("hpo.config.search_space.config must be a mapping.")
        search_space_plugin = SearchSpaceFactory(search_space_name=search_space_type).build(config=dict(search_space_cfg))
        if not isinstance(search_space_plugin, BaseSearchSpace):
            raise TypeError(
                "Search-space plugin "
                f"'{search_space_type}' must build BaseSearchSpace, got {type(search_space_plugin).__name__}."
            )

        serialized_search_space = search_space_cfg.get("search_space", {})
        if not isinstance(serialized_search_space, Mapping):
            raise TypeError("hpo.config.search_space.config.search_space must be a mapping of serialized parameters.")

        super().__init__(
            objective=objective_plugin,
            search_space=search_space_plugin,
            search_space_config=dict(serialized_search_space),
            enabled=enabled,
            n_trials=n_trials,
            direction=direction,
            seed=seed,
            study_name=study_name,
            storage=storage,
            fallback_dir=fallback_dir,
            allow_schema_fallback=allow_schema_fallback,
            loader_split_seed_mode=loader_split_seed_mode,
            loader_split_seed=loader_split_seed,
            max_train_batches=max_train_batches,
            max_val_batches=max_val_batches,
        )
