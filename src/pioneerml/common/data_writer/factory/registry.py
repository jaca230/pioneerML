from __future__ import annotations

from collections.abc import Callable

_WRITER_REGISTRY: dict[str, type] = {}


def register_writer(name: str) -> Callable[[type], type]:
    key = str(name).strip()
    if not key:
        raise ValueError("Writer registry name must be non-empty.")

    def _decorator(writer_cls: type) -> type:
        existing = _WRITER_REGISTRY.get(key)
        if existing is not None and existing is not writer_cls:
            raise ValueError(
                f"Writer registry entry '{key}' already exists for {existing.__name__}; "
                f"cannot re-register with {writer_cls.__name__}."
            )
        _WRITER_REGISTRY[key] = writer_cls
        return writer_cls

    return _decorator


def resolve_writer(name: str) -> type:
    key = str(name).strip()
    writer_cls = _WRITER_REGISTRY.get(key)
    if writer_cls is None:
        known = ", ".join(sorted(_WRITER_REGISTRY)) or "<empty>"
        raise KeyError(f"Unknown writer '{key}'. Registered writers: {known}")
    return writer_cls


def list_registered_writers() -> tuple[str, ...]:
    return tuple(sorted(_WRITER_REGISTRY))

