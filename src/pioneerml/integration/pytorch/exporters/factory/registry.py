from __future__ import annotations

from pioneerml.plugin import NamespacedPluginRegistry

from ..base_exporter import BaseExporter

REGISTRY = NamespacedPluginRegistry[type[BaseExporter]](
    namespace="exporter",
    expected_type=BaseExporter,
    label="Exporter plugin",
)

