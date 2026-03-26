from __future__ import annotations

from pioneerml.plugin import NamespacedPluginRegistry

from ..base_search_space import BaseSearchSpace

REGISTRY = NamespacedPluginRegistry[type[BaseSearchSpace]](
    namespace="search_space",
    expected_type=BaseSearchSpace,
    label="Search-space plugin",
)
