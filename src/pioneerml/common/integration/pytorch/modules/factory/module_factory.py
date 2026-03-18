from __future__ import annotations

from typing import Any

from .registry import resolve_module


class ModuleFactory:
    def __init__(
        self,
        *,
        module_cls: type | None = None,
        module_name: str | None = None,
    ) -> None:
        if module_cls is None and module_name is None:
            raise ValueError("ModuleFactory requires either module_cls or module_name.")
        self.module_cls = module_cls
        self.module_name = None if module_name is None else str(module_name).strip()

    def _resolve_module_class(self) -> type:
        if self.module_cls is not None:
            return self.module_cls
        if self.module_name is None:
            raise RuntimeError("ModuleFactory has neither module_cls nor module_name.")
        return resolve_module(self.module_name)

    def build_module(self, *, model, module_params: dict[str, Any] | None = None):
        module_cls = self._resolve_module_class()
        params = dict(module_params or {})
        if hasattr(module_cls, "from_factory"):
            return module_cls.from_factory(model=model, config=params)
        return module_cls(model=model, **params)

