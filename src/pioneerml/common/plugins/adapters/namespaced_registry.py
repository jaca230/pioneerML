from __future__ import annotations

from collections.abc import Mapping
from typing import Generic, TypeVar

from ..api import list_plugins, register_plugin, resolve_plugin

T = TypeVar("T")


class PluginMappingView(Mapping[str, T]):
    def __init__(self, *, registry: "NamespacedPluginRegistry[T]") -> None:
        self._registry = registry

    def __getitem__(self, key: str) -> T:
        return self._registry.resolve(key)

    def __iter__(self):
        yield from self._registry.list()

    def __len__(self) -> int:
        return len(self._registry.list())


class NamespacedPluginRegistry(Generic[T]):
    """Thin typed wrapper over the global plugin manager for a fixed namespace."""

    def __init__(
        self,
        *,
        namespace: str,
        expected_type: type | None = None,
        class_base: type | None = None,
        label: str = "Plugin",
    ) -> None:
        self.namespace = str(namespace)
        self.expected_type = expected_type
        self.class_base = class_base
        self.label = str(label)

    def register(self, name: str):
        return register_plugin(namespace=self.namespace, name=str(name))

    def resolve(self, name: str) -> T:
        value = resolve_plugin(namespace=self.namespace, name=str(name))
        if self.expected_type is not None:
            if isinstance(value, type) and isinstance(self.expected_type, type):
                if not issubclass(value, self.expected_type):
                    raise TypeError(
                        f"{self.label} '{name}' in namespace '{self.namespace}' must be subclass of "
                        f"{self.expected_type.__name__}, got {value.__name__}."
                    )
            elif not isinstance(value, self.expected_type):
                raise TypeError(
                    f"{self.label} '{name}' in namespace '{self.namespace}' must be "
                    f"{self.expected_type.__name__}, got {type(value).__name__}."
                )
        if self.class_base is not None:
            if not isinstance(value, type):
                raise TypeError(
                    f"{self.label} '{name}' in namespace '{self.namespace}' must resolve to a class, "
                    f"got {type(value).__name__}."
                )
            if not issubclass(value, self.class_base):
                raise TypeError(
                    f"{self.label} '{name}' in namespace '{self.namespace}' must subclass "
                    f"{self.class_base.__name__}, got {value.__name__}."
                )
        return value

    def list(self) -> tuple[str, ...]:
        try:
            return list_plugins(namespace=self.namespace)
        except KeyError:
            return ()

    def mapping_view(self) -> Mapping[str, T]:
        return PluginMappingView(registry=self)
