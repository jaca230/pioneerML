from __future__ import annotations

from typing import Any

from pioneerml.common.plugins import NamespacedPluginFactory


class ArchitectureFactory(NamespacedPluginFactory[Any]):
    def __init__(
        self,
        *,
        architecture_cls: type | None = None,
        architecture_name: str | None = None,
    ) -> None:
        super().__init__(
            namespace="architecture",
            plugin_cls=architecture_cls,
            plugin_name=architecture_name,
            expected_instance_type=None,
            label="Architecture",
        )
