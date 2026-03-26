from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from pioneerml.plugin import NamespacedPluginFactory

from ..base_output_backend import OutputBackend


class OutputBackendFactory(NamespacedPluginFactory[OutputBackend]):
    def __init__(
        self,
        *,
        backend_cls: type[OutputBackend] | None = None,
        backend_name: str | None = None,
        config: Mapping[str, Any] | None = None,
    ) -> None:
        super().__init__(
            namespace="output_backend",
            plugin_cls=backend_cls,
            plugin_name=backend_name,
            expected_instance_type=OutputBackend,
            label="Output backend",
            base_config=dict(config or {}),
        )

