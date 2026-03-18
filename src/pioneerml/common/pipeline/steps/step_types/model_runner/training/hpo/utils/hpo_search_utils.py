from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import optuna

from pioneerml.common.integration.optuna.manager import OptunaStudyManager


def suggest_range(cfg: Mapping, key: str, *, default_low: float, default_high: float):
    raw = cfg.get(key)
    if isinstance(raw, Mapping):
        low = float(raw.get("low", default_low))
        high = float(raw.get("high", default_high))
        log = bool(raw.get("log", True))
        return low, high, log
    if isinstance(raw, (list, tuple)) and len(raw) == 2:
        return float(raw[0]), float(raw[1]), True
    return float(default_low), float(default_high), True


def build_hpo_trainer_kwargs(cfg: Mapping) -> dict:
    kwargs = dict(cfg.get("trainer_kwargs") or {})
    max_train_batches = cfg.get("max_train_batches")
    if max_train_batches is not None:
        kwargs.setdefault("limit_train_batches", int(max_train_batches))
    max_val_batches = cfg.get("max_val_batches")
    if max_val_batches is not None:
        kwargs.setdefault("limit_val_batches", int(max_val_batches))
    return kwargs


def resolve_batch_size_for_trial(
    *,
    trial: optuna.Trial,
    fixed_batch_size: int | None,
    min_exp: int,
    max_exp: int,
) -> int:
    if fixed_batch_size is not None:
        return int(fixed_batch_size)
    return int(1 << int(trial.suggest_int("batch_size_exp", min_exp, max_exp)))


def best_batch_size(*, study, fixed_batch_size: int | None, min_exp: int) -> int:
    if fixed_batch_size is not None:
        return int(fixed_batch_size)
    return int(1 << int(study.best_params.get("batch_size_exp", min_exp)))


def optimize_study(
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


def resolve_loader_split_seed_for_trial(
    *,
    cfg: Mapping[str, Any],
    trial_number: int,
) -> int | None:
    mode = str(cfg.get("loader_split_seed_mode", "fixed")).strip().lower()
    if mode not in {"fixed", "per_trial", "random"}:
        raise ValueError(
            "loader_split_seed_mode must be one of: ['fixed', 'per_trial', 'random'] "
            f"(got '{mode}')."
        )
    if mode == "random":
        return None

    raw_seed = cfg.get("loader_split_seed", cfg.get("seed"))
    if raw_seed in (None, "", "none", "None"):
        return None
    base_seed = int(raw_seed)
    if mode == "fixed":
        return base_seed
    return int(base_seed + int(trial_number))


def with_trial_loader_split_seed(
    *,
    cfg: Mapping[str, Any],
    trial_number: int,
) -> dict[str, Any]:
    mode = str(cfg.get("loader_split_seed_mode", "fixed")).strip().lower()
    if mode not in {"fixed", "per_trial", "random"}:
        raise ValueError(
            "loader_split_seed_mode must be one of: ['fixed', 'per_trial', 'random'] "
            f"(got '{mode}')."
        )
    split_seed = resolve_loader_split_seed_for_trial(
        cfg=cfg,
        trial_number=trial_number,
    )
    out = dict(cfg)
    raw_loader_cfg = dict(out.get("loader_config") or {})
    has_scoped = any(k in raw_loader_cfg for k in ("base", "train", "val", "evaluate", "export", "inference"))

    if has_scoped:
        base_cfg = dict(raw_loader_cfg.get("base") or {})
        if split_seed is None and mode == "random":
            base_cfg.pop("split_seed", None)
        elif split_seed is not None:
            base_cfg["split_seed"] = int(split_seed)
        raw_loader_cfg["base"] = base_cfg
    else:
        if split_seed is None and mode == "random":
            raw_loader_cfg.pop("split_seed", None)
        elif split_seed is not None:
            raw_loader_cfg["split_seed"] = int(split_seed)
    out["loader_config"] = raw_loader_cfg
    return out
