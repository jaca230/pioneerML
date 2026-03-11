from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping
from typing import Any

import pytorch_lightning as pl


class BaseFactoryEarlyStopping(pl.callbacks.EarlyStopping, ABC):
    """Factory-capable early stopping base."""

    TYPE: str = "base"
    _COMMON_DEFAULTS = {
        "monitor": "val_loss",
        "mode": "min",
        "patience": 5,
        "strict": True,
        "check_finite": True,
        "verbose": False,
    }

    @classmethod
    def common_kwargs_from_config(cls, *, config: Mapping[str, Any]) -> dict[str, Any]:
        out = dict(cls._COMMON_DEFAULTS)
        out.update({k: config[k] for k in cls._COMMON_DEFAULTS if k in config})
        return out

    @classmethod
    @abstractmethod
    def from_factory(cls, *, config: Mapping[str, Any] | None = None) -> pl.callbacks.EarlyStopping:
        raise NotImplementedError
