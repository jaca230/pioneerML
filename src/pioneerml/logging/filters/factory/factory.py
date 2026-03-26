from __future__ import annotations

from pioneerml.plugin import NamespacedPluginFactory

from ..base_log_filter import BaseLogFilter


class LogFilterFactory(NamespacedPluginFactory[BaseLogFilter]):
    def __init__(
        self,
        *,
        log_filter_cls: type[BaseLogFilter] | None = None,
        log_filter_name: str | None = None,
    ) -> None:
        super().__init__(
            namespace="log_filter",
            plugin_cls=log_filter_cls,
            plugin_name=log_filter_name,
            expected_instance_type=BaseLogFilter,
            label="Log filter",
        )

