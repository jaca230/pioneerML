from __future__ import annotations

from pioneerml.plugin import NamespacedPluginRegistry

from ..base_data_writer import BaseDataWriter

REGISTRY = NamespacedPluginRegistry[type[BaseDataWriter]](
    namespace="writer",
    expected_type=BaseDataWriter,
    label="Writer plugin",
)
