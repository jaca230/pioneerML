from __future__ import annotations

from pioneerml.plugin import NamespacedPluginRegistry

from ..base_hpo import BaseHPO

REGISTRY = NamespacedPluginRegistry[type[BaseHPO]](
    namespace="hpo",
    expected_type=BaseHPO,
    label="HPO plugin",
)
