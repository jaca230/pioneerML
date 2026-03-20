from __future__ import annotations

from pioneerml.common.plugins import NamespacedPluginRegistry

from ..base_hpo import BaseHPO

REGISTRY = NamespacedPluginRegistry[type[BaseHPO]](
    namespace="hpo",
    expected_type=BaseHPO,
    label="HPO plugin",
)
