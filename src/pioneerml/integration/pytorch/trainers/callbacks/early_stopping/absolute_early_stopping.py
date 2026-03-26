from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import pytorch_lightning as pl

from .base_early_stopping import BaseFactoryEarlyStopping


class AbsoluteEarlyStopping(BaseFactoryEarlyStopping):
    TYPE = "absolute"

    @classmethod
    def from_factory(cls, *, config: Mapping[str, Any] | None = None) -> pl.callbacks.EarlyStopping:
        resolved = dict(config or {})
        kwargs = cls.common_kwargs_from_config(config=resolved)
        min_delta = float(resolved.get("min_delta", 0.0))
        return cls(min_delta=min_delta, **kwargs)
