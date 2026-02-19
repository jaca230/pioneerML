from __future__ import annotations

import optuna

from pioneerml.common.loader import PionStopGraphLoader
from pioneerml.common.models.regressors import PionStopRegressor
from pioneerml.common.pipeline.objective.base import BaseObjectiveAdapter
from pioneerml.common.pipeline.services import suggest_range
from pioneerml.common.pipeline.services.training.utils import GraphLightningModule, QuantilePinballLoss


class PionStopObjectiveAdapter(BaseObjectiveAdapter):
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
            "in_channels": int(PionStopGraphLoader.NODE_FEATURE_DIM),
            "group_prob_dimension": int(PionStopGraphLoader.NUM_CLASSES),
            "splitter_prob_dimension": int(PionStopGraphLoader.NUM_CLASSES),
            "endpoint_pred_dimension": int(PionStopGraphLoader.ENDPOINT_DIM),
            "event_affinity_dimension": int(PionStopGraphLoader.EVENT_AFFINITY_DIM),
            "pion_stop_pred_dimension": 0,
            "hidden": int(hidden),
            "heads": int(heads),
            "layers": int(trial.suggest_int("layers", int(layers_low), int(layers_high))),
            "dropout": float(trial.suggest_float("dropout", float(drop_low), float(drop_high))),
            "output_dim": int(PionStopGraphLoader.TARGET_DIM),
        }

    def build_hpo_module_train_cfg(self, *, cfg: dict, train_params: dict) -> dict:
        out = {
            "scheduler_step_size": (cfg.get("train") or {}).get("scheduler_step_size"),
            "scheduler_gamma": float((cfg.get("train") or {}).get("scheduler_gamma", 0.5)),
        }
        out.update(dict(train_params or {}))
        return out

    def default_model_cfg(self, model_cfg: dict | None = None) -> dict:
        cfg = dict(model_cfg or {})
        cfg.setdefault("in_channels", int(PionStopGraphLoader.NODE_FEATURE_DIM))
        cfg.setdefault("group_prob_dimension", int(PionStopGraphLoader.NUM_CLASSES))
        cfg.setdefault("splitter_prob_dimension", int(PionStopGraphLoader.NUM_CLASSES))
        cfg.setdefault("endpoint_pred_dimension", int(PionStopGraphLoader.ENDPOINT_DIM))
        cfg.setdefault("event_affinity_dimension", int(PionStopGraphLoader.EVENT_AFFINITY_DIM))
        cfg.setdefault("pion_stop_pred_dimension", 0)
        cfg.setdefault("hidden", 192)
        cfg.setdefault("heads", 4)
        cfg.setdefault("layers", 3)
        cfg.setdefault("dropout", 0.1)
        cfg.setdefault("output_dim", int(PionStopGraphLoader.TARGET_DIM))
        return cfg

    def build_model(self, *, model_cfg: dict, compile_cfg: dict | None, context: str):
        _ = compile_cfg
        _ = context
        cfg = self.default_model_cfg(model_cfg)
        hidden = int(cfg.get("hidden", 192))
        heads = int(cfg.get("heads", 4))
        if hidden % heads != 0:
            raise ValueError(f"hidden ({hidden}) must be divisible by heads ({heads})")
        return PionStopRegressor(
            in_channels=int(cfg["in_channels"]),
            group_prob_dimension=int(cfg["group_prob_dimension"]),
            splitter_prob_dimension=int(cfg["splitter_prob_dimension"]),
            endpoint_pred_dimension=int(cfg["endpoint_pred_dimension"]),
            event_affinity_dimension=int(cfg["event_affinity_dimension"]),
            pion_stop_pred_dimension=int(cfg["pion_stop_pred_dimension"]),
            hidden=hidden,
            heads=heads,
            layers=int(cfg.get("layers", 3)),
            dropout=float(cfg.get("dropout", 0.1)),
            output_dim=int(cfg["output_dim"]),
        )

    def build_module(self, *, model, train_cfg: dict):
        return GraphLightningModule(
            model,
            task="regression",
            loss_fn=QuantilePinballLoss(num_outputs=3, quantiles=(0.16, 0.50, 0.84)),
            lr=float(train_cfg["lr"]),
            weight_decay=float(train_cfg["weight_decay"]),
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
                "in_channels": int(PionStopGraphLoader.NODE_FEATURE_DIM),
                "group_prob_dimension": int(PionStopGraphLoader.NUM_CLASSES),
                "splitter_prob_dimension": int(PionStopGraphLoader.NUM_CLASSES),
                "endpoint_pred_dimension": int(PionStopGraphLoader.ENDPOINT_DIM),
                "event_affinity_dimension": int(PionStopGraphLoader.EVENT_AFFINITY_DIM),
                "pion_stop_pred_dimension": 0,
                "hidden": int(study.best_params["hidden"]),
                "heads": int(study.best_params["heads"]),
                "layers": int(study.best_params["layers"]),
                "dropout": float(study.best_params["dropout"]),
                "output_dim": int(PionStopGraphLoader.TARGET_DIM),
            },
        }
