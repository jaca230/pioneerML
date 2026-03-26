from __future__ import annotations

from pioneerml.plugin import NamespacedPluginRegistry

from .base_model_handle import BaseModelHandle

REGISTRY = NamespacedPluginRegistry[type[BaseModelHandle]](
    namespace="model_handle",
    expected_type=BaseModelHandle,
    label="Model handle plugin",
)
