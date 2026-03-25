from __future__ import annotations

from pioneerml.common.plugins import NamespacedPluginRegistry

from ..backends.base_backend import InputBackend

REGISTRY = NamespacedPluginRegistry[type[InputBackend]](
    namespace="input_backend",
    expected_type=InputBackend,
    label="Input backend plugin",
)

