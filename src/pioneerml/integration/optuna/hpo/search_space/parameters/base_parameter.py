from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import optuna


class BaseSearchParameter(ABC):
    @abstractmethod
    def suggest(self, *, trial: optuna.Trial, name: str) -> Any:
        raise NotImplementedError
