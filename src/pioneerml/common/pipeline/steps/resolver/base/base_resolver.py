from __future__ import annotations

from abc import ABC
from typing import Any


class BaseResolver(ABC):
    def __init__(self, *, step: Any) -> None:
        self.step = step
