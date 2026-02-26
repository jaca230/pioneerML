from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import MutableMapping
from typing import Any


class BaseStage(ABC):
    """Contract for composable loader stages."""

    name: str = "base"
    requires: tuple[str, ...] = ()
    provides: tuple[str, ...] = ()

    def validate(self, state: MutableMapping[str, Any]) -> None:
        missing = [k for k in self.requires if k not in state]
        if missing:
            raise RuntimeError(f"Stage '{self.name}' missing required state keys: {missing}")

    @abstractmethod
    def run(self, *, state: MutableMapping[str, Any], loader) -> None:
        raise NotImplementedError
