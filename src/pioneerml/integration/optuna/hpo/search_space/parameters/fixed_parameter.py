from __future__ import annotations

from typing import Any

import optuna

from .base_parameter import BaseSearchParameter
from .factory.registry import REGISTRY as SEARCH_PARAMETER_REGISTRY


@SEARCH_PARAMETER_REGISTRY.register("fixed")
class FixedSearchParameter(BaseSearchParameter):
    def __init__(self, *, value: Any) -> None:
        self.value = value

    def suggest(self, *, trial: optuna.Trial, name: str) -> Any:
        _ = trial, name
        return self.value
