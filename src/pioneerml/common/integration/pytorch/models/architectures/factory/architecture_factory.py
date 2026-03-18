from __future__ import annotations

from typing import Any

from .registry import resolve_architecture


class ArchitectureFactory:
    def __init__(
        self,
        *,
        architecture_cls: type | None = None,
        architecture_name: str | None = None,
    ) -> None:
        if architecture_cls is None and architecture_name is None:
            raise ValueError("ArchitectureFactory requires either architecture_cls or architecture_name.")
        self.architecture_cls = architecture_cls
        self.architecture_name = None if architecture_name is None else str(architecture_name).strip()

    def _resolve_architecture_class(self) -> type:
        if self.architecture_cls is not None:
            return self.architecture_cls
        if self.architecture_name is None:
            raise RuntimeError("ArchitectureFactory has neither architecture_cls nor architecture_name.")
        return resolve_architecture(self.architecture_name)

    def build_architecture(self, *, architecture_params: dict[str, Any] | None = None):
        architecture_cls = self._resolve_architecture_class()
        params = dict(architecture_params or {})
        if hasattr(architecture_cls, "from_factory"):
            return architecture_cls.from_factory(config=params)
        return architecture_cls(**params)

