from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from .adapters.builders import DefaultPluginBuilder
from .factory import PluginFactory
from .manager import PluginManager

PLUGIN_MANAGER = PluginManager()
PLUGIN_FACTORY = PluginFactory(
    manager=PLUGIN_MANAGER,
    default_builder=DefaultPluginBuilder(),
)


def register_plugin(*, namespace: str, name: str):
    return PLUGIN_MANAGER.register(namespace=namespace, name=name)


def register_plugin_value(*, namespace: str, name: str, plugin: Any) -> None:
    PLUGIN_MANAGER.register_value(namespace=namespace, name=name, plugin=plugin)


def resolve_plugin(*, namespace: str, name: str) -> Any:
    return PLUGIN_MANAGER.resolve(namespace=namespace, name=name)


def list_plugins(*, namespace: str) -> tuple[str, ...]:
    return PLUGIN_MANAGER.list(namespace=namespace)


def build_plugin(
    *,
    namespace: str,
    name: str,
    config: Mapping[str, Any] | None = None,
) -> Any:
    return PLUGIN_FACTORY.build(
        namespace=namespace,
        name=name,
        config=config,
    )
