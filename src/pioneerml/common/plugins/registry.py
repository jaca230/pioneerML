from __future__ import annotations

from collections.abc import Callable
import inspect
from threading import RLock
from typing import Generic, TypeVar

T = TypeVar("T")


def normalize_identifier(value: str, *, label: str) -> str:
    out = str(value).strip().lower()
    if out == "":
        raise ValueError(f"{label} must be non-empty.")
    return out


class PluginRegistry(Generic[T]):
    """Typed namespace-local plugin registry."""

    def __init__(self, *, namespace: str) -> None:
        self.namespace = normalize_identifier(namespace, label="Plugin namespace")
        self._entries: dict[str, T] = {}
        self._lock = RLock()

    def register(self, name: str) -> Callable[[T], T]:
        key = normalize_identifier(name, label=f"{self.namespace} plugin name")

        def _decorator(plugin: T) -> T:
            self.register_value(name=key, plugin=plugin)
            return plugin

        return _decorator

    def register_value(self, *, name: str, plugin: T) -> None:
        key = normalize_identifier(name, label=f"{self.namespace} plugin name")
        with self._lock:
            existing = self._entries.get(key)
            if existing is not None and existing is not plugin:
                # Allow idempotent re-registration when symbol/source matches.
                if inspect.isclass(existing) and inspect.isclass(plugin):
                    same_symbol = (
                        getattr(existing, "__module__", None) == getattr(plugin, "__module__", None)
                        and getattr(existing, "__qualname__", None) == getattr(plugin, "__qualname__", None)
                    )
                    same_source = inspect.getsourcefile(existing) == inspect.getsourcefile(plugin)
                    if same_symbol or same_source:
                        self._entries[key] = existing
                        return
                raise ValueError(
                    f"Plugin '{key}' already registered in namespace '{self.namespace}' by "
                    f"{type(existing).__name__}; cannot re-register with {type(plugin).__name__}."
                )
            self._entries[key] = plugin

    def resolve(self, name: str) -> T:
        key = normalize_identifier(name, label=f"{self.namespace} plugin name")
        with self._lock:
            plugin = self._entries.get(key)
            if plugin is None:
                known = ", ".join(sorted(self._entries)) or "<empty>"
                raise KeyError(
                    f"Unknown plugin '{key}' in namespace '{self.namespace}'. Registered plugins: {known}"
                )
            return plugin

    def list(self) -> tuple[str, ...]:
        with self._lock:
            return tuple(sorted(self._entries))

    def mapping(self) -> dict[str, T]:
        with self._lock:
            return dict(self._entries)
