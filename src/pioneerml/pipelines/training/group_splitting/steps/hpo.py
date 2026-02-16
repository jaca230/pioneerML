from collections.abc import Mapping

import optuna
import torch.nn as nn
from zenml import step

from pioneerml.common.loader import GroupSplitterGraphLoader
from pioneerml.common.models.classifiers import GroupSplitter
from pioneerml.common.optuna.manager import OptunaStudyManager
from pioneerml.common.training.lightning import GraphLightningModule
from pioneerml.pipelines.training.group_splitting.dataset import GroupSplitterDataset
from pioneerml.pipelines.training.group_splitting.steps.config import resolve_step_config
from pioneerml.pipelines.training.group_splitting.steps.train import (
    _apply_lightning_warnings_filter,
    _fit_with_loaders,
    _merge_config,
)
from pioneerml.common.training import maybe_compile_model


def _suggest_range(cfg: Mapping, key: str, *, default_low: float, default_high: float):
    raw = cfg.get(key)
    if isinstance(raw, Mapping):
        low = float(raw.get("low", default_low))
        high = float(raw.get("high", default_high))
        log = bool(raw.get("log", True))
        return low, high, log
    if isinstance(raw, (list, tuple)) and len(raw) == 2:
        return float(raw[0]), float(raw[1]), True
    return float(default_low), float(default_high), True


def _select_objective_value(module) -> float:
    if module.val_epoch_loss_history:
        return float(module.val_epoch_loss_history[-1])
    if module.val_loss_history:
        return float(module.val_loss_history[-1])
    if module.train_epoch_loss_history:
        return float(module.train_epoch_loss_history[-1])
    if module.train_loss_history:
        return float(module.train_loss_history[-1])
    return float("inf")


def _resolve_batch_size_search(cfg: Mapping) -> tuple[int | None, int, int]:
    raw = cfg.get("batch_size", {"min_exp": 5, "max_exp": 7})
    if isinstance(raw, Mapping):
        min_exp = int(raw.get("min_exp", 0))
        max_exp = int(raw.get("max_exp", 2))
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


def _build_hpo_trainer_kwargs(cfg: Mapping) -> dict:
    kwargs = dict(cfg.get("trainer_kwargs") or {})
    max_train_batches = cfg.get("max_train_batches")
    if max_train_batches is not None:
        kwargs.setdefault("limit_train_batches", int(max_train_batches))
    max_val_batches = cfg.get("max_val_batches")
    if max_val_batches is not None:
        kwargs.setdefault("limit_val_batches", int(max_val_batches))
    return kwargs


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


@step
def tune_group_splitter(
    dataset: GroupSplitterDataset,
    pipeline_config: dict | None = None,
) -> dict:
    step_config = resolve_step_config(pipeline_config, "hpo")
    if step_config is None or (isinstance(step_config, dict) and step_config.get("enabled") is False):
        return {}

    _apply_lightning_warnings_filter()

    defaults = {
        "n_trials": 5,
        "max_epochs": 3,
        "lr": {"low": 1e-4, "high": 1e-2, "log": True},
        "weight_decay": {"low": 1e-6, "high": 1e-3, "log": True},
        "grad_clip": 2.0,
        "scheduler_step_size": 2,
        "scheduler_gamma": 0.5,
        "threshold": 0.5,
        "trainer_kwargs": {"enable_progress_bar": True},
        "batch_size": {"min_exp": 5, "max_exp": 7},
        "shuffle": True,
        "chunk_row_groups": 4,
        "chunk_workers": 0,
        "max_train_batches": None,
        "max_val_batches": None,
        "early_stopping": {
            "enabled": False,
            "monitor": "val_loss",
            "mode": "min",
            "patience": 3,
            "min_delta": 0.0,
            "min_delta_mode": "absolute",
        },
        "compile": {"enabled": False, "mode": "default"},
        "direction": "minimize",
        "seed": None,
        "study_name": "group_splitter_hpo",
        "storage": None,
        "fallback_dir": None,
        "allow_schema_fallback": True,
        "model": {
            "hidden": {"low": 64, "high": 256, "log": False},
            "heads": {"low": 2, "high": 8, "log": False},
            "layers": {"low": 1, "high": 4, "log": False},
            "dropout": {"low": 0.0, "high": 0.3, "log": False},
        },
    }
    cfg = _merge_config(defaults, step_config)
    if isinstance(step_config, Mapping) and "data_fraction" in step_config:
        raise ValueError(
            "hpo.data_fraction has been removed. "
            "Use loader.config_json.sample_fraction (with deterministic split settings) instead."
        )

    lr_low, lr_high, lr_log = _suggest_range(cfg, "lr", default_low=1e-4, default_high=1e-2)
    wd_low, wd_high, wd_log = _suggest_range(cfg, "weight_decay", default_low=1e-6, default_high=1e-3)

    base_model_cfg = dict(cfg.get("model") or {})
    if "in_channels" not in base_model_cfg:
        base_model_cfg["in_channels"] = int(dataset.data.x.shape[-1])
    if "prob_dimension" not in base_model_cfg:
        base_model_cfg["prob_dimension"] = int(dataset.data.group_probs.shape[-1])
    if "layers" not in base_model_cfg and "num_blocks" in base_model_cfg:
        base_model_cfg["layers"] = int(base_model_cfg["num_blocks"])

    fixed_batch_size, min_batch_size_exp, max_batch_size_exp = _resolve_batch_size_search(cfg)
    base_loader = getattr(dataset, "loader", None)
    if not isinstance(base_loader, GroupSplitterGraphLoader):
        raise RuntimeError("Dataset is missing GroupSplitterGraphLoader required for chunked HPO.")
    if not base_loader.include_targets:
        raise RuntimeError("GroupSplitterGraphLoader must run in train mode for HPO.")
    chunk_row_groups = int(cfg.get("chunk_row_groups", 4))
    chunk_workers = int(cfg.get("chunk_workers", 0))
    trainer_kwargs = _build_hpo_trainer_kwargs(cfg)

    def objective(trial: optuna.Trial) -> float:
        lr = trial.suggest_float("lr", lr_low, lr_high, log=lr_log)
        weight_decay = trial.suggest_float("weight_decay", wd_low, wd_high, log=wd_log)

        if fixed_batch_size is not None:
            batch_size = fixed_batch_size
        else:
            batch_size_exp = trial.suggest_int("batch_size_exp", min_batch_size_exp, max_batch_size_exp)
            batch_size = int(1 << int(batch_size_exp))

        hidden_low, hidden_high, _ = _suggest_range(base_model_cfg, "hidden", default_low=64, default_high=256)
        heads_low, heads_high, _ = _suggest_range(base_model_cfg, "heads", default_low=2, default_high=8)
        layers_low, layers_high, _ = _suggest_range(base_model_cfg, "layers", default_low=1, default_high=4)
        drop_low, drop_high, _ = _suggest_range(base_model_cfg, "dropout", default_low=0.0, default_high=0.3)

        heads = trial.suggest_int("heads", int(heads_low), int(heads_high))
        hidden_low_i = int(hidden_low)
        hidden_high_i = int(hidden_high)
        hidden_low_adj = ((hidden_low_i + heads - 1) // heads) * heads
        hidden_high_adj = (hidden_high_i // heads) * heads
        if hidden_low_adj > hidden_high_adj:
            hidden = hidden_low_adj
        else:
            hidden = trial.suggest_int("hidden", hidden_low_adj, hidden_high_adj, step=heads)

        layers = trial.suggest_int("layers", int(layers_low), int(layers_high))
        dropout = trial.suggest_float("dropout", float(drop_low), float(drop_high))

        model = GroupSplitter(
            in_channels=int(base_model_cfg["in_channels"]),
            prob_dimension=int(base_model_cfg["prob_dimension"]),
            hidden=hidden,
            heads=heads,
            layers=layers,
            dropout=float(dropout),
            num_classes=int(dataset.targets.shape[-1]),
        )
        model = maybe_compile_model(model, cfg.get("compile"), context="tune_group_splitter")
        module = GraphLightningModule(
            model,
            task="classification",
            loss_fn=nn.BCEWithLogitsLoss(),
            lr=lr,
            weight_decay=weight_decay,
            threshold=float(cfg["threshold"]),
            scheduler_step_size=int(cfg["scheduler_step_size"]) if cfg.get("scheduler_step_size") is not None else None,
            scheduler_gamma=float(cfg["scheduler_gamma"]),
        )

        loader_provider = base_loader.with_runtime(
            batch_size=batch_size,
            row_groups_per_chunk=chunk_row_groups,
            num_workers=chunk_workers,
        )
        train_loader = loader_provider.make_dataloader(shuffle_batches=bool(cfg.get("shuffle", True)))
        val_loader = loader_provider.make_dataloader(shuffle_batches=False)
        _fit_with_loaders(
            module=module,
            train_loader=train_loader,
            val_loader=val_loader,
            max_epochs=int(cfg["max_epochs"]),
            grad_clip=float(cfg["grad_clip"]) if cfg.get("grad_clip") is not None else None,
            trainer_kwargs=trainer_kwargs,
            early_stopping_cfg=dict(cfg.get("early_stopping") or {}),
        )
        return _select_objective_value(module)

    sampler = None
    if cfg.get("seed") is not None:
        sampler = optuna.samplers.TPESampler(seed=int(cfg["seed"]))

    study, storage_used = _optimize(
        objective=objective,
        n_trials=int(cfg["n_trials"]),
        study_name=str(cfg.get("study_name", "group_splitter_hpo")),
        direction=str(cfg["direction"]),
        storage=cfg.get("storage"),
        fallback_dir=cfg.get("fallback_dir"),
        allow_schema_fallback=bool(cfg.get("allow_schema_fallback", True)),
        sampler=sampler,
    )

    return {
        "lr": float(study.best_params["lr"]),
        "weight_decay": float(study.best_params["weight_decay"]),
        "batch_size": int(
            fixed_batch_size
            if fixed_batch_size is not None
            else 1 << int(study.best_params.get("batch_size_exp", min_batch_size_exp))
        ),
        "study_name": study.study_name,
        "storage": storage_used,
        "model": {
            "in_channels": int(base_model_cfg["in_channels"]),
            "prob_dimension": int(base_model_cfg["prob_dimension"]),
            "hidden": int(study.best_params["hidden"]),
            "heads": int(study.best_params["heads"]),
            "layers": int(study.best_params["layers"]),
            "dropout": float(study.best_params["dropout"]),
        },
    }
