from __future__ import annotations

from pioneerml.plugin import NamespacedPluginFactory

from .base_metric import BaseMetric


class MetricFactory(NamespacedPluginFactory[BaseMetric]):
    def __init__(
        self,
        *,
        metric_cls: type[BaseMetric] | None = None,
        metric_name: str | None = None,
    ) -> None:
        super().__init__(
            namespace="metric",
            plugin_cls=metric_cls,
            plugin_name=metric_name,
            expected_instance_type=BaseMetric,
            label="Metric",
        )
