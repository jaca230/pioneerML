from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from ......resolver import BaseConfigResolver


class EvaluationConfigResolver(BaseConfigResolver):
    def resolve(self, *, cfg: dict[str, Any]) -> None:
        evaluator = cfg.get("evaluator")
        if not isinstance(evaluator, Mapping):
            raise ValueError("evaluation.evaluator must be a mapping with keys ['type', 'config'].")
        evaluator = dict(evaluator)
        evaluator_type = evaluator.get("type")
        if not isinstance(evaluator_type, str) or evaluator_type.strip() == "":
            raise ValueError("evaluation.evaluator.type must be a non-empty string.")
        if str(evaluator_type).strip().lower() == "required":
            raise ValueError("evaluation.evaluator.type must be set to a concrete registered evaluator plugin.")
        evaluator_config = evaluator.get("config", {})
        if not isinstance(evaluator_config, Mapping):
            raise ValueError("evaluation.evaluator.config must be a mapping when provided.")
        cfg["evaluator"] = {"type": str(evaluator_type).strip(), "config": dict(evaluator_config)}
