from __future__ import annotations

from pioneerml.common.data_writer.base_data_writer import BaseDataWriter
from pioneerml.common.plugins import NamespacedPluginFactory


class WriterFactory(NamespacedPluginFactory[BaseDataWriter]):
    def __init__(
        self,
        *,
        writer_cls: type[BaseDataWriter] | None = None,
        writer_name: str | None = None,
        config: dict | None = None,
    ) -> None:
        super().__init__(
            namespace="writer",
            plugin_cls=writer_cls,
            plugin_name=writer_name,
            expected_instance_type=BaseDataWriter,
            label="Writer",
            base_config=config,
        )
