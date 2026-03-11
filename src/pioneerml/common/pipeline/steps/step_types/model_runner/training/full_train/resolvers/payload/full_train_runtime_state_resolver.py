from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from .......resolver import BasePayloadResolver


class FullTrainRuntimeStateResolver(BasePayloadResolver):
    def resolve(
        self,
        *,
        payloads: Mapping[str, Any] | None,
        runtime_state: dict[str, Any],
    ) -> None:
        objective_adapter = getattr(self.step, "objective_adapter", None)
        if objective_adapter is None:
            raise RuntimeError(f"{self.step.__class__.__name__} is missing objective_adapter.")
        runtime_state["objective_adapter"] = objective_adapter
        runtime_state["training_context"] = f"train_{self.step.__class__.__name__.lower()}"
        runtime_state["hpo_params"] = self._resolve_hpo_params(payloads=payloads)
        runtime_state["upstream_payloads"] = dict(payloads or {})

    def _resolve_hpo_params(self, *, payloads: Mapping[str, Any] | None) -> dict[str, Any]:
        if isinstance(payloads, Mapping):
            hpo_payload = payloads.get("hpo") or payloads.get("hpo_payload")
            if isinstance(hpo_payload, Mapping):
                payload_params = hpo_payload.get("hpo_params")
                if isinstance(payload_params, Mapping):
                    return dict(payload_params)
            direct = payloads.get("hpo_params")
            if isinstance(direct, Mapping):
                return dict(direct)

        hpo_payload = getattr(self.step, "hpo_payload", None)
        if isinstance(hpo_payload, Mapping):
            payload_params = hpo_payload.get("hpo_params")
            if isinstance(payload_params, Mapping):
                return dict(payload_params)
        raw = getattr(self.step, "hpo_params", None)
        if raw is None:
            return {}
        if isinstance(raw, Mapping):
            nested = raw.get("hpo_params") if "hpo_params" in raw else None
            if isinstance(nested, Mapping):
                return dict(nested)
            return dict(raw)
        raise TypeError(f"{self.step.__class__.__name__}.hpo_params must be a mapping when provided.")
