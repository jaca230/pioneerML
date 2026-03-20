from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Callable

import optuna

from pioneerml.common.data_loader import BaseLoaderManager
from pioneerml.common.integration.optuna.manager import OptunaStudyManager
from pioneerml.common.pipeline.steps.resolver import BasePayloadResolver


class HPOStateResolver(BasePayloadResolver):
    def resolve(
        self,
        *,
        payloads: Mapping[str, Any] | None,
        runtime_state: dict[str, Any],
    ) -> None:
        _ = payloads
        cfg = dict(self.step.config_json)
        hpo_cfg = dict(runtime_state.get("hpo_config") or {})
        loader_manager = runtime_state.get("loader_manager")
        if not isinstance(loader_manager, BaseLoaderManager):
            raise RuntimeError(f"{self.step.__class__.__name__} runtime_state missing valid 'loader_manager'.")

        fixed_batch_size, min_exp, max_exp = self._resolve_batch_size_search(hpo_cfg)
        runtime_state["hpo_fixed_batch_size"] = fixed_batch_size
        runtime_state["hpo_min_exp"] = min_exp
        runtime_state["hpo_max_exp"] = max_exp
        runtime_state["hpo_loader_builder"] = self._make_loader_builder(loader_manager=loader_manager)
        runtime_state["hpo_batch_size_for_trial"] = self._make_batch_size_for_trial_builder(
            fixed_batch_size=fixed_batch_size,
            min_exp=min_exp,
            max_exp=max_exp,
        )
        runtime_state["hpo_best_batch_size"] = self._make_best_batch_size_builder(
            fixed_batch_size=fixed_batch_size,
            min_exp=min_exp,
        )
        runtime_state["hpo_trial_cfg_builder"] = self._make_trial_cfg_builder(cfg=cfg, hpo_cfg=hpo_cfg)
        runtime_state["hpo_trainer_kwargs_builder"] = self._make_trainer_kwargs_builder(hpo_cfg=hpo_cfg)
        runtime_state["hpo_study_optimizer"] = self._make_study_optimizer()
        runtime_state["hpo_result_builder"] = self._make_result_builder()

    @classmethod
    def _make_loader_builder(cls, *, loader_manager: BaseLoaderManager) -> Callable[..., tuple[Any, Any]]:
        def _build(*, cfg: dict[str, Any], batch_size: int):
            manager_cls = type(loader_manager)
            trial_manager_cfg = dict(loader_manager.config)
            raw_trial_cfg = dict(dict(cfg.get("loader_manager") or {}).get("config") or {})
            if raw_trial_cfg:
                trial_manager_cfg.update(raw_trial_cfg)
            per_trial_manager = manager_cls(
                config=trial_manager_cfg
            )
            train_provider, train_params = per_trial_manager.build_provider(
                purpose="train",
                forced_batch_size=batch_size,
            )
            val_provider, val_params = per_trial_manager.build_provider(
                purpose="val",
                forced_batch_size=batch_size,
            )
            if not train_provider.include_targets or not val_provider.include_targets:
                raise RuntimeError("HPO expects train/val loaders with targets enabled.")

            train_loader = train_provider.make_dataloader(shuffle_batches=bool(train_params.get("shuffle_batches", True)))
            val_loader = val_provider.make_dataloader(shuffle_batches=bool(val_params.get("shuffle_batches", False)))
            return train_loader, val_loader

        return _build

    @staticmethod
    def _resolve_batch_size_search(cfg: Mapping[str, Any], *, default_min_exp: int = 5, default_max_exp: int = 7):
        raw = cfg.get("batch_size", {"min_exp": default_min_exp, "max_exp": default_max_exp})
        if isinstance(raw, Mapping):
            min_exp = int(raw.get("min_exp", default_min_exp))
            max_exp = int(raw.get("max_exp", default_max_exp))
            if min_exp > max_exp:
                min_exp, max_exp = max_exp, min_exp
            return None, min_exp, max_exp
        if isinstance(raw, (list, tuple)):
            values = [int(v) for v in raw if int(v) > 0]
            if not values:
                return 1, 0, 0
            if len(values) == 1:
                return values[0], 0, 0
            min_value = min(values)
            max_value = max(values)
            min_exp = int(max(min_value - 1, 0)).bit_length()
            max_exp = int(max_value).bit_length() - 1
            if min_exp > max_exp:
                return min_value, 0, 0
            return None, min_exp, max_exp
        fixed = int(raw)
        return fixed, 0, 0

    @staticmethod
    def _make_batch_size_for_trial_builder(
        *,
        fixed_batch_size: int | None,
        min_exp: int,
        max_exp: int,
    ) -> Callable[[optuna.Trial], int]:
        def _build(trial: optuna.Trial) -> int:
            if fixed_batch_size is not None:
                return int(fixed_batch_size)
            return int(1 << int(trial.suggest_int("batch_size_exp", min_exp, max_exp)))

        return _build

    @staticmethod
    def _make_best_batch_size_builder(
        *,
        fixed_batch_size: int | None,
        min_exp: int,
    ) -> Callable[[Any], int]:
        def _build(study: Any) -> int:
            if fixed_batch_size is not None:
                return int(fixed_batch_size)
            return int(1 << int(study.best_params.get("batch_size_exp", min_exp)))

        return _build

    @classmethod
    def _make_trial_cfg_builder(
        cls,
        *,
        cfg: Mapping[str, Any],
        hpo_cfg: Mapping[str, Any],
    ) -> Callable[[int], dict[str, Any]]:
        cfg_dict = dict(cfg)
        hpo_cfg_dict = dict(hpo_cfg)

        def _build(trial_number: int) -> dict[str, Any]:
            mode = str(hpo_cfg_dict.get("loader_split_seed_mode", "fixed")).strip().lower()
            if mode not in {"fixed", "per_trial", "random"}:
                raise ValueError(
                    "loader_split_seed_mode must be one of: ['fixed', 'per_trial', 'random'] "
                    f"(got '{mode}')."
                )
            split_seed = cls._resolve_loader_split_seed_for_trial(
                hpo_cfg=hpo_cfg_dict,
                trial_number=trial_number,
            )
            out = dict(cfg_dict)
            loader_manager_block = dict(out.get("loader_manager") or {})
            loader_manager_cfg = dict(loader_manager_block.get("config") or {})
            defaults_block = dict(loader_manager_cfg.get("defaults") or {})
            defaults_cfg = dict(defaults_block.get("config") or {})
            if split_seed is None and mode == "random":
                defaults_cfg.pop("split_seed", None)
            elif split_seed is not None:
                defaults_cfg["split_seed"] = int(split_seed)
            defaults_block["config"] = defaults_cfg
            loader_manager_cfg["defaults"] = defaults_block
            loader_manager_block["config"] = loader_manager_cfg
            out["loader_manager"] = loader_manager_block
            return out

        return _build

    @staticmethod
    def _resolve_loader_split_seed_for_trial(
        *,
        hpo_cfg: Mapping[str, Any],
        trial_number: int,
    ) -> int | None:
        mode = str(hpo_cfg.get("loader_split_seed_mode", "fixed")).strip().lower()
        if mode not in {"fixed", "per_trial", "random"}:
            raise ValueError(
                "loader_split_seed_mode must be one of: ['fixed', 'per_trial', 'random'] "
                f"(got '{mode}')."
            )
        if mode == "random":
            return None
        raw_seed = hpo_cfg.get("loader_split_seed", hpo_cfg.get("seed"))
        if raw_seed in (None, "", "none", "None"):
            return None
        base_seed = int(raw_seed)
        if mode == "fixed":
            return base_seed
        return int(base_seed + int(trial_number))

    @staticmethod
    def _make_trainer_kwargs_builder(*, hpo_cfg: Mapping[str, Any]) -> Callable[[Mapping[str, Any]], dict[str, Any]]:
        hpo_cfg_dict = dict(hpo_cfg)

        def _build(cfg: Mapping[str, Any]) -> dict[str, Any]:
            trainer_cfg = dict(cfg.get("trainer") or {})
            trainer_inner_cfg = dict(trainer_cfg.get("config") or {})
            kwargs = dict(trainer_inner_cfg.get("trainer_kwargs") or {})
            max_train_batches = hpo_cfg_dict.get("max_train_batches")
            if max_train_batches is not None:
                kwargs.setdefault("limit_train_batches", int(max_train_batches))
            max_val_batches = hpo_cfg_dict.get("max_val_batches")
            if max_val_batches is not None:
                kwargs.setdefault("limit_val_batches", int(max_val_batches))
            return kwargs

        return _build

    @staticmethod
    def _make_study_optimizer() -> Callable[..., tuple[Any, str | None]]:
        def _optimize(
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

        return _optimize

    @staticmethod
    def _make_result_builder() -> Callable[..., dict[str, Any]]:
        def _build(
            *,
            study,
            storage_used: str | None,
            batch_size: int,
        ) -> dict[str, Any]:
            best_params = dict(study.best_params)
            best_params["batch_size"] = int(batch_size)
            return {
                "study_name": study.study_name,
                "storage": storage_used,
                "best_params": best_params,
            }

        return _build
