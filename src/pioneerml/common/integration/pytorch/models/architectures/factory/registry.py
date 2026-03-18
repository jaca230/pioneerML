from __future__ import annotations

from collections.abc import Callable
import inspect

_ARCHITECTURE_REGISTRY: dict[str, type] = {}


def register_architecture(name: str) -> Callable[[type], type]:
    key = str(name).strip()
    if not key:
        raise ValueError("Architecture registry name must be non-empty.")

    def _decorator(architecture_cls: type) -> type:
        existing = _ARCHITECTURE_REGISTRY.get(key)
        if existing is not None and existing is not architecture_cls:
            same_symbol = (
                getattr(existing, "__module__", None) == getattr(architecture_cls, "__module__", None)
                and getattr(existing, "__qualname__", None) == getattr(architecture_cls, "__qualname__", None)
            )
            same_source = inspect.getsourcefile(existing) == inspect.getsourcefile(architecture_cls)
            if same_symbol or same_source:
                _ARCHITECTURE_REGISTRY[key] = existing
                return architecture_cls
            raise ValueError(
                f"Architecture registry entry '{key}' already exists for {existing.__name__}; "
                f"cannot re-register with {architecture_cls.__name__}."
            )
        _ARCHITECTURE_REGISTRY[key] = architecture_cls
        return architecture_cls

    return _decorator


def resolve_architecture(name: str) -> type:
    key = str(name).strip()
    architecture_cls = _ARCHITECTURE_REGISTRY.get(key)
    if architecture_cls is None:
        known = ", ".join(sorted(_ARCHITECTURE_REGISTRY)) or "<empty>"
        raise KeyError(f"Unknown architecture '{key}'. Registered architectures: {known}")
    return architecture_cls


def list_registered_architectures() -> tuple[str, ...]:
    return tuple(sorted(_ARCHITECTURE_REGISTRY))

