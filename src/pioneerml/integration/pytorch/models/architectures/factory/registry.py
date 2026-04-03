from __future__ import annotations

from pioneerml.plugin import NamespacedPluginRegistry

from ..base_architecture import BaseArchitecture

REGISTRY = NamespacedPluginRegistry[type[BaseArchitecture]](
    namespace="architecture",
    expected_type=BaseArchitecture,
    label="Architecture plugin",
)
