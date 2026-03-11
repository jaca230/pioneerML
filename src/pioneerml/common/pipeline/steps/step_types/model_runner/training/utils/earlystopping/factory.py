from __future__ import annotations

from collections.abc import Mapping

import pytorch_lightning as pl

from .absolute_early_stopping import AbsoluteEarlyStopping
from .base_early_stopping import BaseFactoryEarlyStopping
from .relative_early_stopping import RelativeEarlyStopping

_EARLY_STOPPING_TYPES: dict[str, type[BaseFactoryEarlyStopping]] = {
    "absolute": AbsoluteEarlyStopping,
    "default": AbsoluteEarlyStopping,
    "relative": RelativeEarlyStopping,
    "percent": RelativeEarlyStopping,
    "pct": RelativeEarlyStopping,
}


def build_early_stopping_callback(*, early_stopping_cfg: Mapping[str, object] | None) -> pl.callbacks.EarlyStopping | None:
    cfg = dict(early_stopping_cfg or {})
    if not bool(cfg.get("enabled", False)):
        return None

    callback_type = str(cfg.get("type", "relative")).strip().lower()
    config = dict(cfg.get("config") or {})
    callback_cls = _EARLY_STOPPING_TYPES.get(callback_type)
    if callback_cls is not None:
        return callback_cls.from_factory(config=config)

    allowed = "absolute, relative"
    raise ValueError(f"Unsupported early_stopping.type '{callback_type}'. Expected one of: [{allowed}]")
