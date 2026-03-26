from __future__ import annotations

from pioneerml.plugin import NamespacedPluginFactory

from ..base_search_space import BaseSearchSpace


class SearchSpaceFactory(NamespacedPluginFactory[BaseSearchSpace]):
    def __init__(
        self,
        *,
        search_space_cls: type[BaseSearchSpace] | None = None,
        search_space_name: str | None = None,
    ) -> None:
        super().__init__(
            namespace="search_space",
            plugin_cls=search_space_cls,
            plugin_name=search_space_name,
            expected_instance_type=BaseSearchSpace,
            label="Search-space",
        )
