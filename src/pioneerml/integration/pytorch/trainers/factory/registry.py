from __future__ import annotations

import pytorch_lightning as pl

from pioneerml.plugin import NamespacedPluginRegistry

REGISTRY = NamespacedPluginRegistry[type[pl.Trainer]](
    namespace="trainer",
    expected_type=pl.Trainer,
    label="Trainer plugin",
)
