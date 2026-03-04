from __future__ import annotations

from collections.abc import Callable

_LOADER_REGISTRY: dict[str, type] = {}


def register_loader(name: str) -> Callable[[type], type]:
    key = str(name).strip()
    if not key:
        raise ValueError("Loader registry name must be non-empty.")

    def _decorator(loader_cls: type) -> type:
        existing = _LOADER_REGISTRY.get(key)
        if existing is not None and existing is not loader_cls:
            raise ValueError(
                f"Loader registry entry '{key}' already exists for {existing.__name__}; "
                f"cannot re-register with {loader_cls.__name__}."
            )
        _LOADER_REGISTRY[key] = loader_cls
        return loader_cls

    return _decorator


def resolve_loader(name: str) -> type:
    key = str(name).strip()
    loader_cls = _LOADER_REGISTRY.get(key)
    if loader_cls is None:
        known = ", ".join(sorted(_LOADER_REGISTRY)) or "<empty>"
        raise KeyError(f"Unknown loader '{key}'. Registered loaders: {known}")
    return loader_cls


def list_registered_loaders() -> tuple[str, ...]:
    return tuple(sorted(_LOADER_REGISTRY))
