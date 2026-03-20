from __future__ import annotations

from pioneerml.common.plugins import NamespacedPluginRegistry

from ..base_parameter import BaseSearchParameter

REGISTRY = NamespacedPluginRegistry[type[BaseSearchParameter]](
    namespace="search_parameter",
    expected_type=BaseSearchParameter,
    label="Search-parameter plugin",
)
