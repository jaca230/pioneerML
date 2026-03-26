from __future__ import annotations

from pioneerml.plugin import NamespacedPluginFactory

from .base_model_handle import BaseModelHandle


class ModelHandleFactory(NamespacedPluginFactory[BaseModelHandle]):
    def __init__(
        self,
        *,
        model_handle_cls: type[BaseModelHandle] | None = None,
        model_type: str | None = None,
    ) -> None:
        super().__init__(
            namespace="model_handle",
            plugin_cls=model_handle_cls,
            plugin_name=model_type,
            expected_instance_type=BaseModelHandle,
            label="Model handle",
        )
