from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping
from pathlib import Path
from typing import Callable, ClassVar

import torch


class BaseModelHandle(ABC):
    """Serializable model-handle contract for deferred model loading."""

    TYPE: str = "base"
    ALIASES: tuple[str, ...] = ()
    _REGISTRY: ClassVar[dict[str, type["BaseModelHandle"]]] = {}

    @classmethod
    def _register_handle_class(cls, handle_cls: type["BaseModelHandle"]) -> type["BaseModelHandle"]:
        type_name = str(getattr(handle_cls, "TYPE", "")).strip().lower()
        if type_name in {"", "base"}:
            raise RuntimeError(
                f"{handle_cls.__name__} must define TYPE with a non-empty non-'base' value to register."
            )
        keys = (type_name, *tuple(str(v).strip().lower() for v in getattr(handle_cls, "ALIASES", ())))
        for key in keys:
            if not key:
                continue
            existing = cls._REGISTRY.get(key)
            if existing is not None and existing is not handle_cls:
                raise RuntimeError(
                    f"Model handle alias '{key}' already registered by {existing.__name__}; "
                    f"cannot re-register with {handle_cls.__name__}."
                )
            cls._REGISTRY[key] = handle_cls
        return handle_cls

    @classmethod
    def register(cls) -> Callable[[type["BaseModelHandle"]], type["BaseModelHandle"]]:
        def decorator(handle_cls: type["BaseModelHandle"]) -> type["BaseModelHandle"]:
            return cls._register_handle_class(handle_cls)

        return decorator

    def __init__(self, *, model_path: str) -> None:
        self.model_path = str(model_path)

    @property
    def path(self) -> Path:
        return Path(self.model_path).expanduser().resolve()

    @abstractmethod
    def load(self, *, device: torch.device):
        """Materialize and return an inference-ready model object."""
        raise NotImplementedError

    def to_payload(self) -> dict[str, str]:
        return {"type": str(self.TYPE), "model_path": str(self.path)}

    @classmethod
    def registry(cls) -> Mapping[str, type["BaseModelHandle"]]:
        return dict(cls._REGISTRY)
