from __future__ import annotations

from typing import Any

import optuna

from .base_parameter import BaseSearchParameter
from .factory.registry import REGISTRY as SEARCH_PARAMETER_REGISTRY


@SEARCH_PARAMETER_REGISTRY.register("categorical")
class CategoricalSearchParameter(BaseSearchParameter):
    def __init__(self, *, choices: list[Any]) -> None:
        self.choices = list(choices)

    def suggest(self, *, trial: optuna.Trial, name: str) -> Any:
        return trial.suggest_categorical(name, self.choices)
