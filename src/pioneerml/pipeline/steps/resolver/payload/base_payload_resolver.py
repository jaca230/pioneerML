from __future__ import annotations

from abc import abstractmethod
from collections.abc import Mapping
from typing import Any

from ..base.base_resolver import BaseResolver


class BasePayloadResolver(BaseResolver):
    @abstractmethod
    def resolve(
        self,
        *,
        payloads: Mapping[str, Any] | None,
        runtime_state: dict[str, Any],
    ) -> None:
        raise NotImplementedError
