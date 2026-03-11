from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from pioneerml.common.data_loader import LoaderFactory

from .......resolver import BasePayloadResolver


class EvaluationRuntimeStateResolver(BasePayloadResolver):
    def resolve(
        self,
        *,
        payloads: Mapping[str, Any] | None,
        runtime_state: dict[str, Any],
    ) -> None:
        runtime_state["module"] = self._resolve_module(payloads=payloads)
        runtime_state["loader_factory"] = self._resolve_loader_factory(payloads=payloads)
        runtime_state["upstream_payloads"] = dict(payloads or {})

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

    def _resolve_loader_factory(self, *, payloads: Mapping[str, Any] | None) -> LoaderFactory:
        if isinstance(payloads, Mapping):
            for key in ("loader", "loader_payload", "dataset"):
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
