from __future__ import annotations

import optuna

from .base_parameter import BaseSearchParameter
from .factory.registry import REGISTRY as SEARCH_PARAMETER_REGISTRY


@SEARCH_PARAMETER_REGISTRY.register("exp2_int")
@SEARCH_PARAMETER_REGISTRY.register("exponent_int")
class ExponentIntSearchParameter(BaseSearchParameter):
    def __init__(
        self,
        *,
        min_exp: int,
        max_exp: int,
        base: int | float = 2,
    ) -> None:
        lo = int(min_exp)
        hi = int(max_exp)
        if lo > hi:
            lo, hi = hi, lo
        self.min_exp = lo
        self.max_exp = hi
        self.base = float(base)
        if self.base <= 0.0 or self.base == 1.0:
            raise ValueError("exponent_int parameter requires base > 0 and base != 1.")

    def suggest(self, *, trial: optuna.Trial, name: str) -> int:
        exp = int(trial.suggest_int(f"{name}_exp", int(self.min_exp), int(self.max_exp)))
        return int(round(self.base ** exp))

