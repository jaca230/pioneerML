from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from pioneerml.common.data_loader import (
    BaseLoaderManager,
    LoaderFactory,
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
        loader_factory = self._resolve_loader_factory(payloads=payloads)
        self._validate_loader_type(loader_factory=loader_factory)
        loader_manager = self._build_loader_manager(loader_factory=loader_factory)
        runtime_state["loader_factory"] = loader_factory
        runtime_state["loader_manager"] = loader_manager
        runtime_state["upstream_payloads"] = dict(payloads or {})

    def _resolve_loader_factory(self, *, payloads: Mapping[str, Any] | None) -> LoaderFactory:
        if isinstance(payloads, Mapping):
            for key in ("loader", "loader_payload", "loader_factory_init", "dataset"):
                candidate = payloads.get(key)
                if isinstance(candidate, Mapping):
                    payload_factory = candidate.get("loader_factory")
                    if isinstance(payload_factory, LoaderFactory):
                        return payload_factory
                if isinstance(candidate, LoaderFactory):
                    return candidate
            payload_factory = payloads.get("loader_factory")
            if isinstance(payload_factory, LoaderFactory):
                return payload_factory
        raise RuntimeError(
            f"{self.step.__class__.__name__} requires upstream payloads containing a loader_factory "
            "(e.g. payloads['loader']['loader_factory'])."
        )

    def _validate_loader_type(self, *, loader_factory: LoaderFactory) -> None:
        manager_block = dict(self.step.config_json.get("loader_manager") or {})
        manager_cfg = dict(manager_block.get("config") or {})
        defaults_cfg = dict(manager_cfg.get("defaults") or {})
        expected_type = str(defaults_cfg.get("type") or "").strip()
        if expected_type == "":
            raise RuntimeError(
                f"{self.step.__class__.__name__} requires non-empty 'loader_manager.config.defaults.type' config."
            )
        actual = str(loader_factory.plugin_name or "").strip()
        if actual == "":
            return
        if actual != expected_type:
            raise RuntimeError(
                f"{self.step.__class__.__name__} loader.type='{expected_type}' does not match upstream "
                f"loader_factory plugin '{actual}'."
            )

    def _build_loader_manager(self, *, loader_factory: LoaderFactory) -> BaseLoaderManager:
        manager_block = dict(self.step.config_json.get("loader_manager") or {})
        manager_type = str(manager_block.get("type") or "").strip()
        if manager_type == "":
            raise RuntimeError(f"{self.step.__class__.__name__} requires non-empty 'loader_manager.type' config.")
        manager_cfg = dict(manager_block.get("config") or {})
        manager_cfg["loader_factory"] = loader_factory
        manager = LoaderManagerFactory(loader_manager_name=manager_type).build(config=manager_cfg)
        if not isinstance(manager, BaseLoaderManager):
            raise RuntimeError(f"{self.step.__class__.__name__} loader_manager factory must return BaseLoaderManager.")
        return manager
