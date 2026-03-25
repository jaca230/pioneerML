from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from pioneerml.common.data_loader import BaseLoaderManager

from ......resolver import BasePayloadResolver


class EvaluationStateResolver(BasePayloadResolver):
    def resolve(
        self,
        *,
        payloads: Mapping[str, Any] | None,
        runtime_state: dict[str, Any],
    ) -> None:
        if not bool(dict(getattr(self.step, "config_json", {}) or {}).get("enabled", True)):
            runtime_state["evaluation_disabled"] = True
            return

        module = self._resolve_module(payloads=payloads)
        loader_manager = runtime_state.get("loader_manager")
        if not isinstance(loader_manager, BaseLoaderManager):
            raise RuntimeError(
                f"{self.step.__class__.__name__} runtime_state missing valid 'loader_manager'."
            )
        provider, loader_params, loader = loader_manager.build_dataloader(
            purpose="test",
            default_shuffle=False,
        )
        if not provider.include_targets:
            raise RuntimeError(f"{self.step.__class__.__name__} expects evaluation loader with targets enabled.")
        runtime_state["module"] = module
        runtime_state["evaluation_provider"] = provider
        runtime_state["evaluation_loader_params"] = loader_params
        runtime_state["evaluation_loader"] = loader

    def _resolve_module(self, *, payloads: Mapping[str, Any] | None):
        if isinstance(payloads, Mapping):
            train_payload = payloads.get("train") or payloads.get("train_payload")
            if isinstance(train_payload, Mapping) and train_payload.get("module") is not None:
                return train_payload.get("module")
            direct = payloads.get("module")
            if direct is not None:
                return direct
        raise RuntimeError(
            f"{self.step.__class__.__name__} requires upstream payloads containing a module "
            "(e.g. payloads['train']['module'] or payloads['module'])."
        )
