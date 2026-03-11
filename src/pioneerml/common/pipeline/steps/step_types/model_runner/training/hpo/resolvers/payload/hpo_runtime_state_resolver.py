from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from ......resolver import BasePayloadResolver


class HPORuntimeStateResolver(BasePayloadResolver):
    def resolve(
        self,
        *,
        payloads: Mapping[str, Any] | None,
        runtime_state: dict[str, Any],
    ) -> None:
        objective_adapter = self.step.build_objective_adapter()
        if objective_adapter is None:
            raise RuntimeError(f"{self.step.__class__.__name__}.build_objective_adapter() returned None.")
        runtime_state["objective_adapter"] = objective_adapter
        runtime_state["hpo_enabled"] = bool(dict(self.step.config_json.get("hpo_config") or {}).get("enabled", True))
        runtime_state["training_context"] = f"tune_{self.step.__class__.__name__.lower()}"
        runtime_state["upstream_payloads"] = dict(payloads or {})
