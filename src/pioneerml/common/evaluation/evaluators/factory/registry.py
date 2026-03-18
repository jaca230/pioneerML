from __future__ import annotations

from collections.abc import Callable
import inspect

_EVALUATOR_REGISTRY: dict[str, type] = {}


def register_evaluator(name: str) -> Callable[[type], type]:
    key = str(name).strip()
    if not key:
        raise ValueError("Evaluator registry name must be non-empty.")

    def _decorator(evaluator_cls: type) -> type:
        existing = _EVALUATOR_REGISTRY.get(key)
        if existing is not None and existing is not evaluator_cls:
            same_symbol = (
                getattr(existing, "__module__", None) == getattr(evaluator_cls, "__module__", None)
                and getattr(existing, "__qualname__", None) == getattr(evaluator_cls, "__qualname__", None)
            )
            same_source = inspect.getsourcefile(existing) == inspect.getsourcefile(evaluator_cls)
            if same_symbol or same_source:
                _EVALUATOR_REGISTRY[key] = existing
                return evaluator_cls
            raise ValueError(
                f"Evaluator registry entry '{key}' already exists for {existing.__name__}; "
                f"cannot re-register with {evaluator_cls.__name__}."
            )
        _EVALUATOR_REGISTRY[key] = evaluator_cls
        return evaluator_cls

    return _decorator


def resolve_evaluator(name: str) -> type:
    key = str(name).strip()
    evaluator_cls = _EVALUATOR_REGISTRY.get(key)
    if evaluator_cls is None:
        known = ", ".join(sorted(_EVALUATOR_REGISTRY)) or "<empty>"
        raise KeyError(f"Unknown evaluator '{key}'. Registered evaluators: {known}")
    return evaluator_cls


def list_registered_evaluators() -> tuple[str, ...]:
    return tuple(sorted(_EVALUATOR_REGISTRY))
