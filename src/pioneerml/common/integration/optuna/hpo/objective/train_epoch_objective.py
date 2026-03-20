from __future__ import annotations

from .base_objective import BaseObjective
from .factory.registry import REGISTRY as OBJECTIVE_REGISTRY


@OBJECTIVE_REGISTRY.register("train_epoch")
class TrainEpochObjective(BaseObjective):
    def objective_from_module(self, module) -> float:
        values = getattr(module, "train_epoch_loss_history", None)
        if isinstance(values, list) and values:
            return float(values[-1])
        return float("inf")
