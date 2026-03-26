from __future__ import annotations

from typing import Any

from pioneerml.integration.optuna import (
    BaseHPO,
    HPOFactory,
)
from pioneerml.pipeline.steps.resolver import BaseConfigResolver


class HPOConfigResolver(BaseConfigResolver):
    def resolve(self, *, cfg: dict[str, Any]) -> None:
        hpo_block = cfg.get("hpo")
        if not isinstance(hpo_block, dict):
            raise TypeError("training.hpo.hpo must be a dict with keys ['type', 'config'].")
        hpo_type = hpo_block.get("type")
        if not isinstance(hpo_type, str) or hpo_type.strip() == "":
            raise ValueError("training.hpo.hpo.type must be a non-empty string.")
        hpo_cfg = hpo_block.get("config")
        if not isinstance(hpo_cfg, dict):
            raise TypeError("training.hpo.hpo.config must be a dict.")

        hpo_plugin = HPOFactory(hpo_name=str(hpo_type).strip()).build(config=dict(hpo_cfg))
        if not isinstance(hpo_plugin, BaseHPO):
            raise RuntimeError(f"{self.step.__class__.__name__} hpo factory must return BaseHPO.")

        cfg["hpo"] = {
            "type": str(hpo_type).strip(),
            "config": dict(hpo_cfg),
        }
        self.step.runtime_state["hpo"] = hpo_plugin
        self.step.runtime_state["hpo_config"] = hpo_plugin.runtime_config()
        self.step.runtime_state["objective"] = hpo_plugin.objective
        self.step.runtime_state["search_space"] = hpo_plugin.search_space
        self.step.runtime_state["hpo_enabled"] = bool(hpo_plugin.enabled)
        self.step.runtime_state["training_context"] = f"tune_{self.step.__class__.__name__.lower()}"
