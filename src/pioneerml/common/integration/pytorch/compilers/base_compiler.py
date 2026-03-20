from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping
from typing import Any


class BaseCompiler(ABC):
    """Base compiler contract for model materialization/transforms."""

    def __init__(self, *, config: Mapping[str, Any] | None = None) -> None:
        self.config: dict[str, Any] = dict(config or {})

    @abstractmethod
    def compile(
        self,
        *,
        model: Any,
        context: str = "run",
    ) -> Any:
        raise NotImplementedError

