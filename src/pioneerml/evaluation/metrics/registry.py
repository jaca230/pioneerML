from __future__ import annotations

from collections.abc import Mapping

from pioneerml.plugin import NamespacedPluginRegistry

from .base_metric import BaseMetric

REGISTRY = NamespacedPluginRegistry[type[BaseMetric]](
    namespace="metric",
    expected_type=BaseMetric,
    label="Metric plugin",
)

METRIC_REGISTRY: Mapping[str, type[BaseMetric]] = REGISTRY.mapping_view()
