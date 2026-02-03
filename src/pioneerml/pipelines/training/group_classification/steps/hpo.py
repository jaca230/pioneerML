from typing import Mapping

import pytorch_lightning as pl
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from zenml import step

from pioneerml.common.models.classifiers import GroupClassifier
from pioneerml.common.optuna.manager import OptunaStudyManager
from pioneerml.common.training.lightning import GraphLightningModule
from pioneerml.common.zenml.utils import detect_available_accelerator

from pioneerml.pipelines.training.group_classification.dataset import GroupClassifierDataset
from pioneerml.pipelines.training.group_classification.steps.config import resolve_step_config
from .train import _apply_lightning_warnings_filter, _merge_config, _split_dataset_to_graphs, _collate_graphs


try:
    import optuna  # type: ignore
except Exception:  # pragma: no cover - optuna optional
    optuna = None


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


@step
def tune_group_classifier(
    dataset: GroupClassifierDataset,
    pipeline_config: dict | None = None,
) -> dict:
    step_config = resolve_step_config(pipeline_config, "hpo")
    if step_config is None or (isinstance(step_config, dict) and step_config.get("enabled") is False):
        return {}

    if optuna is None:  # pragma: no cover - optuna optional
        raise RuntimeError("Optuna is required for HPO but is not installed.")

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
        "batch_size": [1, 2, 4],
        "shuffle": True,
        "direction": "minimize",
        "seed": None,
        "study_name": "group_classifier_hpo",
        "storage": None,
        "fallback_dir": None,
        "allow_schema_fallback": True,
        "model": {
            "hidden": {"low": 64, "high": 256, "log": False},
            "heads": {"low": 2, "high": 8, "log": False},
            "num_blocks": {"low": 1, "high": 4, "log": False},
            "dropout": {"low": 0.0, "high": 0.3, "log": False},
        },
    }
    cfg = _merge_config(defaults, step_config)

    lr_low, lr_high, lr_log = _suggest_range(cfg, "lr", default_low=1e-4, default_high=1e-2)
    wd_low, wd_high, wd_log = _suggest_range(cfg, "weight_decay", default_low=1e-6, default_high=1e-3)
    base_model_cfg = dict(cfg.get("model") or {})
    if "in_dim" not in base_model_cfg:
        base_model_cfg["in_dim"] = int(dataset.data.x.shape[-1])
    if "edge_dim" not in base_model_cfg:
        base_model_cfg["edge_dim"] = int(dataset.data.edge_attr.shape[-1])

    accelerator, devices = detect_available_accelerator()
    graphs = _split_dataset_to_graphs(dataset)
    if not graphs:
        raise RuntimeError("No non-empty graphs found in dataset for HPO.")

    def objective(trial: "optuna.Trial") -> float:
        lr = trial.suggest_float("lr", lr_low, lr_high, log=lr_log)
        weight_decay = trial.suggest_float("weight_decay", wd_low, wd_high, log=wd_log)
        batch_size_cfg = cfg.get("batch_size", [1])
        if isinstance(batch_size_cfg, (list, tuple)) and batch_size_cfg:
            batch_size = int(trial.suggest_categorical("batch_size", list(batch_size_cfg)))
        else:
            batch_size = int(batch_size_cfg)
        hidden_low, hidden_high, _ = _suggest_range(base_model_cfg, "hidden", default_low=64, default_high=256)
        heads_low, heads_high, _ = _suggest_range(base_model_cfg, "heads", default_low=2, default_high=8)
        blocks_low, blocks_high, _ = _suggest_range(base_model_cfg, "num_blocks", default_low=1, default_high=4)
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
        num_blocks = trial.suggest_int("num_blocks", int(blocks_low), int(blocks_high))
        dropout = trial.suggest_float("dropout", float(drop_low), float(drop_high))

        model = GroupClassifier(
            in_dim=int(base_model_cfg["in_dim"]),
            edge_dim=int(base_model_cfg["edge_dim"]),
            hidden=hidden,
            heads=heads,
            num_blocks=num_blocks,
            dropout=float(dropout),
            num_classes=int(dataset.targets.shape[-1]),
        )
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

        trainer_kwargs = dict(cfg.get("trainer_kwargs") or {})
        if cfg.get("grad_clip") is not None:
            trainer_kwargs.setdefault("gradient_clip_val", float(cfg["grad_clip"]))
        lightning_trainer = pl.Trainer(
            accelerator=accelerator,
            devices=devices,
            max_epochs=int(cfg["max_epochs"]),
            enable_checkpointing=False,
            logger=False,
            **trainer_kwargs,
        )
        trial_train_loader = DataLoader(
            graphs,
            batch_size=batch_size,
            shuffle=bool(cfg.get("shuffle", True)),
            collate_fn=_collate_graphs,
        )
        trial_val_loader = DataLoader(
            graphs,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=_collate_graphs,
        )
        lightning_trainer.fit(module, train_dataloaders=trial_train_loader, val_dataloaders=trial_val_loader)

        if module.val_epoch_loss_history:
            return float(module.val_epoch_loss_history[-1])
        if module.val_loss_history:
            return float(module.val_loss_history[-1])
        if module.train_epoch_loss_history:
            return float(module.train_epoch_loss_history[-1])
        if module.train_loss_history:
            return float(module.train_loss_history[-1])
        return float("inf")

    sampler = None
    if cfg.get("seed") is not None:
        sampler = optuna.samplers.TPESampler(seed=int(cfg["seed"]))

    manager = OptunaStudyManager(
        study_name=str(cfg.get("study_name", "group_classifier_hpo")),
        direction=str(cfg["direction"]),
        storage=cfg.get("storage"),
        fallback_dir=cfg.get("fallback_dir"),
        allow_schema_fallback=bool(cfg.get("allow_schema_fallback", True)),
    )
    study, storage_used = manager.create_or_load(sampler=sampler)
    study.optimize(objective, n_trials=int(cfg["n_trials"]))

    return {
        "lr": float(study.best_params["lr"]),
        "weight_decay": float(study.best_params["weight_decay"]),
        "batch_size": int(study.best_params.get("batch_size", 1)),
        "study_name": study.study_name,
        "storage": storage_used,
        "model": {
            "in_dim": int(base_model_cfg["in_dim"]),
            "edge_dim": int(base_model_cfg["edge_dim"]),
            "hidden": int(study.best_params["hidden"]),
            "heads": int(study.best_params["heads"]),
            "num_blocks": int(study.best_params["num_blocks"]),
            "dropout": float(study.best_params["dropout"]),
        },
    }
