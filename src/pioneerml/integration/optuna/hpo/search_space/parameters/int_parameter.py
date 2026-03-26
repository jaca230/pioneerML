from __future__ import annotations

import optuna

from .base_parameter import BaseSearchParameter
from .factory.registry import REGISTRY as SEARCH_PARAMETER_REGISTRY


@SEARCH_PARAMETER_REGISTRY.register("int")
class IntSearchParameter(BaseSearchParameter):
    def __init__(
        self,
        *,
        low: int,
        high: int,
        log: bool = False,
        step: int | None = None,
    ) -> None:
        self.low = int(low)
        self.high = int(high)
        self.log = bool(log)
        self.step = None if step is None else int(step)

    def suggest(self, *, trial: optuna.Trial, name: str) -> int:
        if self.step is None:
            return int(trial.suggest_int(name, self.low, self.high, log=self.log))
        return int(trial.suggest_int(name, self.low, self.high, step=self.step, log=self.log))
