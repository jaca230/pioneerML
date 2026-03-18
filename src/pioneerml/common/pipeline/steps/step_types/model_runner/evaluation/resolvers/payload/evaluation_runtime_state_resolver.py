from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from ......resolver import BasePayloadResolver


class EvaluationRuntimeStateResolver(BasePayloadResolver):
    def resolve(
        self,
        *,
        payloads: Mapping[str, Any] | None,
        runtime_state: dict[str, Any],
    ) -> None:
        runtime_state["module"] = self._resolve_module(payloads=payloads)

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
