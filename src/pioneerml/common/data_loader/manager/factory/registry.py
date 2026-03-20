from __future__ import annotations

from pioneerml.common.plugins import NamespacedPluginRegistry

from ..base_loader_manager import BaseLoaderManager

REGISTRY = NamespacedPluginRegistry[type[BaseLoaderManager]](
    namespace="loader_manager",
    expected_type=BaseLoaderManager,
    label="Loader manager plugin",
)

