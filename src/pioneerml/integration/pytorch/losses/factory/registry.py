from __future__ import annotations

from pioneerml.plugin import NamespacedPluginRegistry

from ..base_loss import BaseLoss

REGISTRY = NamespacedPluginRegistry[type[BaseLoss]](
    namespace="loss",
    expected_type=BaseLoss,
    label="Loss plugin",
)

