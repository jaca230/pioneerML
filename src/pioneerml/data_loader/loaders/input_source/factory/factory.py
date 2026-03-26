from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from pioneerml.plugin import NamespacedPluginFactory

from ..backends.base_backend import InputBackend


class InputBackendFactory(NamespacedPluginFactory[InputBackend]):
    def __init__(
        self,
        *,
        backend_cls: type[InputBackend] | None = None,
        backend_name: str | None = None,
        config: Mapping[str, Any] | None = None,
    ) -> None:
        super().__init__(
            namespace="input_backend",
            plugin_cls=backend_cls,
            plugin_name=backend_name,
            expected_instance_type=InputBackend,
            label="Input backend",
            base_config=dict(config or {}),
        )

