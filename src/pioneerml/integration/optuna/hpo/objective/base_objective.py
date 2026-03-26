from __future__ import annotations

from abc import ABC, abstractmethod


class BaseObjective(ABC):
    @abstractmethod
    def objective_from_module(self, module) -> float:
        raise NotImplementedError
