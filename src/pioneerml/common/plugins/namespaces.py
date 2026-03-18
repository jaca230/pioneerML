from __future__ import annotations

from collections.abc import Iterable
import importlib
from importlib import metadata
import inspect
import pkgutil
from typing import Any

from .manager import PluginManager


def bootstrap_modules(*, modules: Iterable[str]) -> list[str]:
    """Import explicit modules so decorator-based registrations execute."""
    loaded: list[str] = []
    for module_name in modules:
        name = str(module_name).strip()
        if name == "":
            continue
        importlib.import_module(name)
        loaded.append(name)
    return loaded


def bootstrap_package(*, package: str, include_private: bool = False) -> list[str]:
    """Recursively import a package tree to execute registration side effects."""
    root = importlib.import_module(str(package))
    if not hasattr(root, "__path__"):
        return [str(root.__name__)]

    loaded: list[str] = [str(root.__name__)]
    prefix = f"{root.__name__}."
    for module_info in pkgutil.walk_packages(root.__path__, prefix=prefix):
        leaf = module_info.name.rsplit(".", 1)[-1]
        if not include_private and leaf.startswith("_"):
            continue
        importlib.import_module(module_info.name)
        loaded.append(module_info.name)
    return loaded


def bootstrap_entrypoints(
    *,
    group: str = "pioneerml.plugins",
    manager: PluginManager | None = None,
) -> list[str]:
    """Load plugin bootstrap callables declared as package entry points.

    Entry point targets may be:
    - a callable with optional `manager` kwarg
    - any importable symbol (loaded for side effects)
    """
    loaded: list[str] = []
    entry_points = metadata.entry_points()
    selected = entry_points.select(group=group) if hasattr(entry_points, "select") else entry_points.get(group, [])
    for ep in selected:
        target = ep.load()
        if callable(target):
            kwargs: dict[str, Any] = {}
            if manager is not None:
                sig = inspect.signature(target)
                if "manager" in sig.parameters:
                    kwargs["manager"] = manager
            target(**kwargs)
        loaded.append(f"{group}:{ep.name}")
    return loaded
