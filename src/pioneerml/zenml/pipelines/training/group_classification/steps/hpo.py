from typing import Mapping

import pytorch_lightning as pl
import torch
import torch.nn as nn
from zenml import step

from pioneerml.models.classifiers import GroupClassifier
from pioneerml.optuna.manager import OptunaStudyManager
from pioneerml.training.lightning import GraphLightningModule
from pioneerml.zenml.utils import detect_available_accelerator

from pioneerml.zenml.pipelines.training.group_classification.batch import GroupClassifierBatch
from .train import _apply_lightning_warnings_filter, _merge_config


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
    batch: GroupClassifierBatch,
    *,
    step_config: dict | None = None,
) -> dict:
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
        "trainer_kwargs": {},
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

    accelerator, devices = detect_available_accelerator()
    trainer = torch.utils.data.DataLoader(
        [batch.data],
        batch_size=1,
        shuffle=False,
        collate_fn=lambda items: items[0],
    )

    def objective(trial: "optuna.Trial") -> float:
        lr = trial.suggest_float("lr", lr_low, lr_high, log=lr_log)
        weight_decay = trial.suggest_float("weight_decay", wd_low, wd_high, log=wd_log)
        model_cfg = dict(cfg.get("model") or {})
        if "in_dim" not in model_cfg:
            model_cfg["in_dim"] = int(batch.data.x.shape[-1])
        if "edge_dim" not in model_cfg:
            model_cfg["edge_dim"] = int(batch.data.edge_attr.shape[-1])
        hidden_low, hidden_high, _ = _suggest_range(model_cfg, "hidden", default_low=64, default_high=256)
        heads_low, heads_high, _ = _suggest_range(model_cfg, "heads", default_low=2, default_high=8)
        blocks_low, blocks_high, _ = _suggest_range(model_cfg, "num_blocks", default_low=1, default_high=4)
        drop_low, drop_high, _ = _suggest_range(model_cfg, "dropout", default_low=0.0, default_high=0.3)

        hidden = trial.suggest_int("hidden", int(hidden_low), int(hidden_high))
        heads = trial.suggest_int("heads", int(heads_low), int(heads_high))
        num_blocks = trial.suggest_int("num_blocks", int(blocks_low), int(blocks_high))
        dropout = trial.suggest_float("dropout", float(drop_low), float(drop_high))

        model = GroupClassifier(
            in_dim=int(model_cfg.get("in_dim", 4)),
            edge_dim=int(model_cfg.get("edge_dim", 4)),
            hidden=hidden,
            heads=heads,
            num_blocks=num_blocks,
            dropout=float(dropout),
            num_classes=int(batch.targets.shape[-1]),
        )
        module = GraphLightningModule(
            model,
            task="classification",
            loss_fn=nn.BCEWithLogitsLoss(),
            lr=lr,
            weight_decay=weight_decay,
        )

        trainer_kwargs = dict(cfg.get("trainer_kwargs") or {})
        lightning_trainer = pl.Trainer(
            accelerator=accelerator,
            devices=devices,
            max_epochs=int(cfg["max_epochs"]),
            enable_checkpointing=False,
            logger=False,
            **trainer_kwargs,
        )
        lightning_trainer.fit(module, train_dataloaders=trainer, val_dataloaders=trainer)

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
        "study_name": study.study_name,
        "storage": storage_used,
        "model": {
            "in_dim": int(model_cfg.get("in_dim", 4)),
            "edge_dim": int(model_cfg.get("edge_dim", 4)),
            "hidden": int(study.best_params["hidden"]),
            "heads": int(study.best_params["heads"]),
            "num_blocks": int(study.best_params["num_blocks"]),
            "dropout": float(study.best_params["dropout"]),
        },
    }
