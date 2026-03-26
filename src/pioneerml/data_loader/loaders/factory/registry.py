from __future__ import annotations

from pioneerml.plugin import NamespacedPluginRegistry

from ..base_loader import BaseLoader

REGISTRY = NamespacedPluginRegistry[type[BaseLoader]](
    namespace="loader",
    expected_type=BaseLoader,
    label="Loader plugin",
)
