from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from pioneerml.common.integration.optuna.objective import BaseObjectiveAdapter

from .......resolver import BaseConfigResolver


class HPORuntimeConfigResolver(BaseConfigResolver):
    def resolve(self, *, cfg: dict[str, Any]) -> None:
        required = ("n_trials", "max_epochs", "loader_config", "batch_size", "direction")
        missing = [k for k in required if k not in cfg]
        if missing:
            raise KeyError(f"training.hpo missing required keys: {missing}")

        objective_adapter = self.step.build_objective_adapter()
        if not isinstance(objective_adapter, BaseObjectiveAdapter):
            raise RuntimeError(
                f"{self.step.__class__.__name__}.build_objective_adapter() must return BaseObjectiveAdapter."
            )
        self.step.runtime_state["objective_adapter"] = objective_adapter
        if "enabled" in cfg:
            hpo_enabled = bool(cfg.get("enabled", True))
        else:
            hpo_enabled = bool(dict(cfg.get("hpo_config") or {}).get("enabled", True))
        self.step.runtime_state["hpo_enabled"] = hpo_enabled
        self.step.runtime_state["training_context"] = f"tune_{self.step.__class__.__name__.lower()}"

    @staticmethod
    def resolve_batch_size_search(cfg: Mapping, *, default_min_exp: int = 5, default_max_exp: int = 7):
        raw = cfg.get("batch_size", {"min_exp": default_min_exp, "max_exp": default_max_exp})
        if isinstance(raw, Mapping):
            min_exp = int(raw.get("min_exp", default_min_exp))
            max_exp = int(raw.get("max_exp", default_max_exp))
            if min_exp > max_exp:
                min_exp, max_exp = max_exp, min_exp
            return None, min_exp, max_exp
        if isinstance(raw, (list, tuple)):
            values = [int(v) for v in raw if int(v) > 0]
            if not values:
                return 1, 0, 0
            if len(values) == 1:
                return values[0], 0, 0
            min_value = min(values)
            max_value = max(values)
            min_exp = int(max(min_value - 1, 0)).bit_length()
            max_exp = int(max_value).bit_length() - 1
            if min_exp > max_exp:
                return min_value, 0, 0
            return None, min_exp, max_exp
        fixed = int(raw)
        return fixed, 0, 0
