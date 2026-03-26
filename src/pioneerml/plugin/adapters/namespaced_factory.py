from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Generic, TypeVar

from ..api import build_plugin
from .builders import DefaultPluginBuilder

T = TypeVar("T")


class NamespacedPluginFactory(Generic[T]):
    """Generic plugin factory wrapper for a fixed namespace."""

    def __init__(
        self,
        *,
        namespace: str,
        plugin_cls: type | None = None,
        plugin_name: str | None = None,
        expected_instance_type: type | None = None,
        label: str = "Plugin",
        base_config: Mapping[str, Any] | None = None,
    ) -> None:
        if plugin_cls is None and plugin_name is None:
            raise ValueError(f"{label} factory requires either plugin_cls or plugin_name.")
        self.namespace = str(namespace)
        self.plugin_cls = plugin_cls
        self.plugin_name = None if plugin_name is None else str(plugin_name).strip()
        self.expected_instance_type = expected_instance_type
        self.label = str(label)
        self.config: dict[str, Any] = dict(base_config or {})
        self._default_builder = DefaultPluginBuilder()

    def build(
        self,
        *,
        config: Mapping[str, Any] | None = None,
    ) -> T:
        cfg = {**dict(self.config), **dict(config or {})}

        if self.plugin_cls is None:
            if self.plugin_name is None:
                raise RuntimeError(f"{self.label} factory has neither plugin_cls nor plugin_name.")
            instance = build_plugin(
                namespace=self.namespace,
                name=str(self.plugin_name),
                config=cfg,
            )
        else:
            instance = self._default_builder.build(
                plugin=self.plugin_cls,
                namespace=self.namespace,
                name=(self.plugin_name or self.plugin_cls.__name__),
                config=cfg,
            )

        if self.expected_instance_type is not None and not isinstance(instance, self.expected_instance_type):
            raise TypeError(
                f"{self.label} factory expected {self.expected_instance_type.__name__} instance, "
                f"got {type(instance).__name__}."
            )
        return instance
