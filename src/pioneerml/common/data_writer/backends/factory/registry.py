from __future__ import annotations

from pioneerml.common.plugins import NamespacedPluginRegistry

from ..base_output_backend import OutputBackend

REGISTRY = NamespacedPluginRegistry[type[OutputBackend]](
    namespace="output_backend",
    expected_type=OutputBackend,
    label="Output backend plugin",
)

