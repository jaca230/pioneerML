from __future__ import annotations

from abc import abstractmethod
from typing import Any

from ..base.base_resolver import BaseResolver


class BaseConfigResolver(BaseResolver):
    @abstractmethod
    def resolve(self, *, cfg: dict[str, Any]) -> None:
        raise NotImplementedError
