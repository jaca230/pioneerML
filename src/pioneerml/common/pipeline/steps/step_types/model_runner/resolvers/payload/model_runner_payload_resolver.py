from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from pioneerml.common.data_loader import LoaderFactory

from .....resolver import BasePayloadResolver


class ModelRunnerPayloadResolver(BasePayloadResolver):
    def resolve(
        self,
        *,
        payloads: Mapping[str, Any] | None,
        runtime_state: dict[str, Any],
    ) -> None:
        runtime_state["loader_factory"] = self._resolve_loader_factory(payloads=payloads)
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
