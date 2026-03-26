from __future__ import annotations

from pioneerml.plugin import NamespacedPluginFactory

from ..base_exporter import BaseExporter


class ExporterFactory(NamespacedPluginFactory[BaseExporter]):
    def __init__(
        self,
        *,
        exporter_cls: type[BaseExporter] | None = None,
        exporter_name: str | None = None,
    ) -> None:
        super().__init__(
            namespace="exporter",
            plugin_cls=exporter_cls,
            plugin_name=exporter_name,
            expected_instance_type=BaseExporter,
            label="Exporter",
        )

