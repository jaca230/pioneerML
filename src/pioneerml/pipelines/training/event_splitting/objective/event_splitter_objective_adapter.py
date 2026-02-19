from __future__ import annotations

import optuna
import torch.nn as nn

from pioneerml.common.loader import EventSplitterGraphLoader
from pioneerml.common.models.components import EventSplitter
from pioneerml.common.pipeline.objective.base import BaseObjectiveAdapter
from pioneerml.common.pipeline.services import suggest_range
from pioneerml.common.pipeline.services.training.utils import GraphLightningModule


class EventSplitterObjectiveAdapter(BaseObjectiveAdapter):
    @staticmethod
    def default_hpo_model_space() -> dict:
        return {
            "hidden": {"low": 96, "high": 320, "log": False},
            "heads": {"low": 2, "high": 8, "log": False},
            "layers": {"low": 1, "high": 4, "log": False},
            "dropout": {"low": 0.0, "high": 0.3, "log": False},
        }

    def suggest_model_params(self, *, trial: optuna.Trial, model_search_cfg: dict | None = None) -> dict:
        cfg = dict(self.default_hpo_model_space())
        if model_search_cfg:
            cfg.update(dict(model_search_cfg))

        hidden_low, hidden_high, _ = suggest_range(cfg, "hidden", default_low=96, default_high=320)
        heads_low, heads_high, _ = suggest_range(cfg, "heads", default_low=2, default_high=8)
        layers_low, layers_high, _ = suggest_range(cfg, "layers", default_low=1, default_high=4)
        drop_low, drop_high, _ = suggest_range(cfg, "dropout", default_low=0.0, default_high=0.3)

        heads = trial.suggest_int("heads", int(heads_low), int(heads_high))
        hidden_low_i = int(hidden_low)
        hidden_high_i = int(hidden_high)
        hidden_low_adj = ((hidden_low_i + heads - 1) // heads) * heads
        hidden_high_adj = (hidden_high_i // heads) * heads
        hidden = (
            hidden_low_adj
            if hidden_low_adj > hidden_high_adj
            else trial.suggest_int("hidden", hidden_low_adj, hidden_high_adj, step=heads)
        )

        return {
            "in_channels": int(EventSplitterGraphLoader.NODE_FEATURE_DIM),
            "group_prob_dimension": int(EventSplitterGraphLoader.NUM_CLASSES),
            "splitter_prob_dimension": int(EventSplitterGraphLoader.NUM_CLASSES),
            "endpoint_dimension": int(EventSplitterGraphLoader.ENDPOINT_DIM),
            "edge_attr_dimension": int(EventSplitterGraphLoader.EDGE_FEATURE_DIM),
            "hidden": int(hidden),
            "heads": int(heads),
            "layers": int(trial.suggest_int("layers", int(layers_low), int(layers_high))),
            "dropout": float(trial.suggest_float("dropout", float(drop_low), float(drop_high))),
        }

    @staticmethod
    def default_hpo_train_space() -> dict:
        return {
            "lr": {"low": 1e-4, "high": 1e-2, "log": True},
            "weight_decay": {"low": 1e-6, "high": 1e-3, "log": True},
        }

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
            "threshold": float((cfg.get("train") or {}).get("threshold", 0.5)),
            "scheduler_step_size": (cfg.get("train") or {}).get("scheduler_step_size"),
            "scheduler_gamma": float((cfg.get("train") or {}).get("scheduler_gamma", 0.5)),
        }
        out.update(dict(train_params or {}))
        return out

    def default_model_cfg(self, model_cfg: dict | None = None) -> dict:
        cfg = dict(model_cfg or {})
        cfg.setdefault("in_channels", int(EventSplitterGraphLoader.NODE_FEATURE_DIM))
        cfg.setdefault("group_prob_dimension", int(EventSplitterGraphLoader.NUM_CLASSES))
        cfg.setdefault("splitter_prob_dimension", int(EventSplitterGraphLoader.NUM_CLASSES))
        cfg.setdefault("endpoint_dimension", int(EventSplitterGraphLoader.ENDPOINT_DIM))
        cfg.setdefault("edge_attr_dimension", int(EventSplitterGraphLoader.EDGE_FEATURE_DIM))
        cfg.setdefault("hidden", 192)
        cfg.setdefault("heads", 4)
        cfg.setdefault("layers", 3)
        cfg.setdefault("dropout", 0.1)
        return cfg

    def build_model(self, *, model_cfg: dict, compile_cfg: dict | None, context: str):
        _ = compile_cfg
        _ = context
        cfg = self.default_model_cfg(model_cfg)
        hidden = int(cfg.get("hidden", 192))
        heads = int(cfg.get("heads", 4))
        if hidden % heads != 0:
            raise ValueError(f"hidden ({hidden}) must be divisible by heads ({heads})")
        return EventSplitter(
            in_channels=int(cfg["in_channels"]),
            group_prob_dimension=int(cfg["group_prob_dimension"]),
            splitter_prob_dimension=int(cfg["splitter_prob_dimension"]),
            endpoint_dimension=int(cfg["endpoint_dimension"]),
            edge_attr_dimension=int(cfg["edge_attr_dimension"]),
            hidden=hidden,
            heads=heads,
            layers=int(cfg.get("layers", 3)),
            dropout=float(cfg.get("dropout", 0.1)),
        )

    def build_module(self, *, model, train_cfg: dict):
        return GraphLightningModule(
            model,
            task="classification",
            loss_fn=nn.BCEWithLogitsLoss(),
            lr=float(train_cfg["lr"]),
            weight_decay=float(train_cfg["weight_decay"]),
            threshold=float(train_cfg.get("threshold", 0.5)),
            scheduler_step_size=(
                int(train_cfg["scheduler_step_size"]) if train_cfg.get("scheduler_step_size") is not None else None
            ),
            scheduler_gamma=float(train_cfg["scheduler_gamma"]),
        )

    def objective_from_module(self, module) -> float:
        if module.val_epoch_loss_history:
            return float(module.val_epoch_loss_history[-1])
        if module.val_loss_history:
            return float(module.val_loss_history[-1])
        if module.train_epoch_loss_history:
            return float(module.train_epoch_loss_history[-1])
        if module.train_loss_history:
            return float(module.train_loss_history[-1])
        return float("inf")

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
            "lr": float(study.best_params["lr"]),
            "weight_decay": float(study.best_params["weight_decay"]),
            "batch_size": int(batch_size),
            "study_name": study.study_name,
            "storage": storage_used,
            "model": {
                "in_channels": int(EventSplitterGraphLoader.NODE_FEATURE_DIM),
                "group_prob_dimension": int(EventSplitterGraphLoader.NUM_CLASSES),
                "splitter_prob_dimension": int(EventSplitterGraphLoader.NUM_CLASSES),
                "endpoint_dimension": int(EventSplitterGraphLoader.ENDPOINT_DIM),
                "edge_attr_dimension": int(EventSplitterGraphLoader.EDGE_FEATURE_DIM),
                "hidden": int(study.best_params["hidden"]),
                "heads": int(study.best_params["heads"]),
                "layers": int(study.best_params["layers"]),
                "dropout": float(study.best_params["dropout"]),
            },
        }
