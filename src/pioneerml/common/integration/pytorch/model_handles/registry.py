from __future__ import annotations

from pioneerml.common.plugins import NamespacedPluginRegistry

from .base_model_handle import BaseModelHandle

REGISTRY = NamespacedPluginRegistry[type[BaseModelHandle]](
    namespace="model_handle",
    expected_type=BaseModelHandle,
    label="Model handle plugin",
)
