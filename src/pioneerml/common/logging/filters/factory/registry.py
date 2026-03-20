from __future__ import annotations

from pioneerml.common.plugins import NamespacedPluginRegistry

from ..base_log_filter import BaseLogFilter

REGISTRY = NamespacedPluginRegistry[type[BaseLogFilter]](
    namespace="log_filter",
    expected_type=BaseLogFilter,
    label="Log filter plugin",
)

LOG_FILTER_REGISTRY = REGISTRY

