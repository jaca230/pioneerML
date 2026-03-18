from __future__ import annotations

from collections.abc import Callable
import inspect

_MODULE_REGISTRY: dict[str, type] = {}


def register_module(name: str) -> Callable[[type], type]:
    key = str(name).strip()
    if not key:
        raise ValueError("Module registry name must be non-empty.")

    def _decorator(module_cls: type) -> type:
        existing = _MODULE_REGISTRY.get(key)
        if existing is not None and existing is not module_cls:
            same_symbol = (
                getattr(existing, "__module__", None) == getattr(module_cls, "__module__", None)
                and getattr(existing, "__qualname__", None) == getattr(module_cls, "__qualname__", None)
            )
            same_source = inspect.getsourcefile(existing) == inspect.getsourcefile(module_cls)
            if same_symbol or same_source:
                _MODULE_REGISTRY[key] = existing
                return module_cls
            raise ValueError(
                f"Module registry entry '{key}' already exists for {existing.__name__}; "
                f"cannot re-register with {module_cls.__name__}."
            )
        _MODULE_REGISTRY[key] = module_cls
        return module_cls

    return _decorator


def resolve_module(name: str) -> type:
    key = str(name).strip()
    module_cls = _MODULE_REGISTRY.get(key)
    if module_cls is None:
        known = ", ".join(sorted(_MODULE_REGISTRY)) or "<empty>"
        raise KeyError(f"Unknown module '{key}'. Registered modules: {known}")
    return module_cls


def list_registered_modules() -> tuple[str, ...]:
    return tuple(sorted(_MODULE_REGISTRY))

