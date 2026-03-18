from __future__ import annotations

from collections.abc import Callable
from threading import RLock
from typing import Any

from .registry import PluginRegistry, normalize_identifier


class PluginManager:
    """Owns all plugin namespaces and their registries."""

    def __init__(self) -> None:
        self._registries: dict[str, PluginRegistry[Any]] = {}
        self._lock = RLock()

    def registry(self, *, namespace: str, create: bool = True) -> PluginRegistry[Any]:
        ns = normalize_identifier(namespace, label="Plugin namespace")
        with self._lock:
            reg = self._registries.get(ns)
            if reg is None:
                if not create:
                    raise KeyError(f"Unknown plugin namespace '{ns}'.")
                reg = PluginRegistry(namespace=ns)
                self._registries[ns] = reg
            return reg

    def register(self, *, namespace: str, name: str) -> Callable[[Any], Any]:
        return self.registry(namespace=namespace).register(name)

    def register_value(self, *, namespace: str, name: str, plugin: Any) -> None:
        self.registry(namespace=namespace).register_value(name=name, plugin=plugin)

    def resolve(self, *, namespace: str, name: str) -> Any:
        return self.registry(namespace=namespace, create=False).resolve(name)

    def list(self, *, namespace: str) -> tuple[str, ...]:
        return self.registry(namespace=namespace, create=False).list()

    def namespaces(self) -> tuple[str, ...]:
        with self._lock:
            return tuple(sorted(self._registries))

    def mappings(self) -> dict[str, dict[str, Any]]:
        with self._lock:
            return {ns: reg.mapping() for ns, reg in self._registries.items()}
