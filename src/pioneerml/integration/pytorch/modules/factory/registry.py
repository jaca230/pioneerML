from __future__ import annotations

import pytorch_lightning as pl

from pioneerml.plugin import NamespacedPluginRegistry

REGISTRY = NamespacedPluginRegistry[type[pl.LightningModule]](
    namespace="module",
    expected_type=pl.LightningModule,
    label="Module plugin",
)
