from __future__ import annotations

from typing import TypeVar

from pioneerml.common.data_loader.loaders.base_loader import BaseLoader
from pioneerml.common.plugins import NamespacedPluginFactory

L = TypeVar("L", bound=BaseLoader)


class LoaderFactory(NamespacedPluginFactory[L]):
    def __init__(
        self,
        *,
        loader_cls: type[L] | None = None,
        loader_name: str | None = None,
        config: dict | None = None,
    ) -> None:
        super().__init__(
            namespace="loader",
            plugin_cls=loader_cls,
            plugin_name=loader_name,
            expected_instance_type=BaseLoader,
            label="Loader",
            base_config=config,
        )
