from __future__ import annotations

from typing import Any

from pioneerml.plugin import NamespacedPluginFactory


class ModuleFactory(NamespacedPluginFactory[Any]):
    def __init__(
        self,
        *,
        module_cls: type | None = None,
        module_name: str | None = None,
    ) -> None:
        super().__init__(
            namespace="module",
            plugin_cls=module_cls,
            plugin_name=module_name,
            expected_instance_type=None,
            label="Module",
        )
