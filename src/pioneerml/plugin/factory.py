from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from .adapters.builders import BasePluginBuilder, DefaultPluginBuilder
from .manager import PluginManager


def _normalize_identifier(value: str, *, label: str) -> str:
    out = str(value).strip().lower()
    if out == "":
        raise ValueError(f"{label} must be non-empty.")
    return out


class PluginFactory:
    """Unified plugin factory with per-namespace builder override support."""

    def __init__(
        self,
        *,
        manager: PluginManager,
        default_builder: BasePluginBuilder | None = None,
        builders_by_namespace: Mapping[str, BasePluginBuilder] | None = None,
    ) -> None:
        self.manager = manager
        self.default_builder: BasePluginBuilder = default_builder or DefaultPluginBuilder()
        self._builders_by_namespace: dict[str, BasePluginBuilder] = {}
        for ns, builder in dict(builders_by_namespace or {}).items():
            self.set_builder(namespace=ns, builder=builder)

    def set_builder(self, *, namespace: str, builder: BasePluginBuilder) -> None:
        ns = _normalize_identifier(namespace, label="Plugin namespace")
        self._builders_by_namespace[ns] = builder

    def builder_for(self, *, namespace: str) -> BasePluginBuilder:
        ns = _normalize_identifier(namespace, label="Plugin namespace")
        return self._builders_by_namespace.get(ns, self.default_builder)

    def build(
        self,
        *,
        namespace: str,
        name: str,
        config: Mapping[str, Any] | None = None,
    ) -> Any:
        ns = _normalize_identifier(namespace, label="Plugin namespace")
        key = _normalize_identifier(name, label=f"{ns} plugin name")
        plugin = self.manager.resolve(namespace=ns, name=key)
        builder = self.builder_for(namespace=ns)
        return builder.build(
            plugin=plugin,
            namespace=ns,
            name=key,
            config=config,
        )
