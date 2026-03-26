from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from pioneerml.data_loader import (
    BaseLoaderManager,
    LoaderManagerFactory,
)

from .....resolver import BasePayloadResolver


class ModelRunnerStateResolver(BasePayloadResolver):
    def resolve(
        self,
        *,
        payloads: Mapping[str, Any] | None,
        runtime_state: dict[str, Any],
    ) -> None:
        loader_manager = self._build_loader_manager()
        runtime_state["loader_factory"] = loader_manager.loader_factory
        runtime_state["loader_manager"] = loader_manager
        runtime_state["upstream_payloads"] = dict(payloads or {})

    def _build_loader_manager(self) -> BaseLoaderManager:
        manager_block = dict(self.step.config_json.get("loader_manager") or {})
        manager_type = str(manager_block.get("type") or "").strip()
        if manager_type == "":
            raise RuntimeError(f"{self.step.__class__.__name__} requires non-empty 'loader_manager.type' config.")
        manager_cfg = dict(manager_block.get("config") or {})
        manager = LoaderManagerFactory(loader_manager_name=manager_type).build(config=manager_cfg)
        if not isinstance(manager, BaseLoaderManager):
            raise RuntimeError(f"{self.step.__class__.__name__} loader_manager factory must return BaseLoaderManager.")
        return manager
