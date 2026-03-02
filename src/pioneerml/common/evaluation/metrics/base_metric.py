from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping
from typing import Any


class BaseMetric(ABC):
    """Base contract for pluggable evaluation metrics."""

    @abstractmethod
    def compute(self, *, context: Mapping[str, Any]) -> dict[str, Any]:
        raise NotImplementedError
