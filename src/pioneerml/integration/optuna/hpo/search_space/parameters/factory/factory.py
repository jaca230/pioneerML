from __future__ import annotations

from pioneerml.plugin import NamespacedPluginFactory

from ..base_parameter import BaseSearchParameter


class SearchParameterFactory(NamespacedPluginFactory[BaseSearchParameter]):
    def __init__(
        self,
        *,
        search_parameter_cls: type[BaseSearchParameter] | None = None,
        search_parameter_name: str | None = None,
    ) -> None:
        super().__init__(
            namespace="search_parameter",
            plugin_cls=search_parameter_cls,
            plugin_name=search_parameter_name,
            expected_instance_type=BaseSearchParameter,
            label="Search-parameter",
        )
