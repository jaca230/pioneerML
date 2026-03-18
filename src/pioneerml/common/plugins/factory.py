from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Protocol

from .builders.default import DefaultPluginBuilder
from .manager import PluginManager
from .registry import normalize_identifier


class PluginBuilder(Protocol):
    def build(
        self,
        *,
        plugin: Any,
        namespace: str,
        name: str,
        config: Mapping[str, Any] | None = None,
        context: Mapping[str, Any] | None = None,
    ) -> Any:
        ...


class PluginFactory:
    """Unified plugin factory with per-namespace builder override support."""

    def __init__(
        self,
        *,
        manager: PluginManager,
        default_builder: PluginBuilder | None = None,
        builders_by_namespace: Mapping[str, PluginBuilder] | None = None,
    ) -> None:
        self.manager = manager
        self.default_builder: PluginBuilder = default_builder or DefaultPluginBuilder()
        self._builders_by_namespace: dict[str, PluginBuilder] = {}
        for ns, builder in dict(builders_by_namespace or {}).items():
            self.set_builder(namespace=ns, builder=builder)

    def set_builder(self, *, namespace: str, builder: PluginBuilder) -> None:
        ns = normalize_identifier(namespace, label="Plugin namespace")
        self._builders_by_namespace[ns] = builder

    def builder_for(self, *, namespace: str) -> PluginBuilder:
        ns = normalize_identifier(namespace, label="Plugin namespace")
        return self._builders_by_namespace.get(ns, self.default_builder)

    def build(
        self,
        *,
        namespace: str,
        name: str,
        config: Mapping[str, Any] | None = None,
        context: Mapping[str, Any] | None = None,
    ) -> Any:
        ns = normalize_identifier(namespace, label="Plugin namespace")
        key = normalize_identifier(name, label=f"{ns} plugin name")
        plugin = self.manager.resolve(namespace=ns, name=key)
        builder = self.builder_for(namespace=ns)
        return builder.build(
            plugin=plugin,
            namespace=ns,
            name=key,
            config=config,
            context=context,
        )
