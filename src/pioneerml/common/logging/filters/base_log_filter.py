from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping
from typing import Any


class BaseLogFilter(ABC):
    def __init__(self, *, config: Mapping[str, Any] | None = None) -> None:
        self.config: dict[str, Any] = dict(config or {})

    @abstractmethod
    def apply(self) -> None:
        raise NotImplementedError

