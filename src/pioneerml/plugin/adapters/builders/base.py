from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping
from typing import Any


class BasePluginBuilder(ABC):
    @abstractmethod
    def build(
        self,
        *,
        plugin: Any,
        namespace: str,
        name: str,
        config: Mapping[str, Any] | None = None,
    ) -> Any:
        raise NotImplementedError
