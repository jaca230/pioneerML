from __future__ import annotations

import optuna

from .base_parameter import BaseSearchParameter
from .factory.registry import REGISTRY as SEARCH_PARAMETER_REGISTRY


@SEARCH_PARAMETER_REGISTRY.register("float")
class FloatSearchParameter(BaseSearchParameter):
    def __init__(
        self,
        *,
        low: float,
        high: float,
        log: bool = False,
        step: float | None = None,
    ) -> None:
        self.low = float(low)
        self.high = float(high)
        self.log = bool(log)
        self.step = None if step is None else float(step)

    def suggest(self, *, trial: optuna.Trial, name: str) -> float:
        if self.step is None:
            return float(trial.suggest_float(name, self.low, self.high, log=self.log))
        return float(trial.suggest_float(name, self.low, self.high, step=self.step, log=self.log))
