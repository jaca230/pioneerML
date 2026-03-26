from __future__ import annotations

from pioneerml.plugin import NamespacedPluginFactory

from ..base_loader_manager import BaseLoaderManager


class LoaderManagerFactory(NamespacedPluginFactory[BaseLoaderManager]):
    def __init__(
        self,
        *,
        loader_manager_cls: type[BaseLoaderManager] | None = None,
        loader_manager_name: str | None = None,
        config: dict | None = None,
    ) -> None:
        super().__init__(
            namespace="loader_manager",
            plugin_cls=loader_manager_cls,
            plugin_name=loader_manager_name,
            expected_instance_type=BaseLoaderManager,
            label="LoaderManager",
            base_config=config,
        )

