from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from ......resolver import BaseConfigResolver


class EvaluationRuntimeConfigResolver(BaseConfigResolver):
    def resolve(self, *, cfg: dict[str, Any]) -> None:
        evaluator_name = cfg.get("evaluator_name")
        if not isinstance(evaluator_name, str):
            raise ValueError("evaluation.evaluator_name must be a string.")
        evaluator_name = evaluator_name.strip()
        if evaluator_name == "":
            raise ValueError("evaluation.evaluator_name cannot be empty.")
        cfg["evaluator_name"] = evaluator_name

        evaluator_config = cfg.get("evaluator_config", {})
        if not isinstance(evaluator_config, Mapping):
            raise ValueError("evaluation.evaluator_config must be a mapping when provided.")
        cfg["evaluator_config"] = dict(evaluator_config)

        cfg["metrics"] = self.resolve_registered_names(
            cfg=cfg,
            config_key="metrics",
            error_context="evaluation.metrics",
        )
        cfg["plots"] = self.resolve_registered_names(
            cfg=cfg,
            config_key="plots",
            error_context="evaluation.plots",
        )
        cfg["default_plot_kwargs"] = self.resolve_default_plot_kwargs(cfg=cfg)
        cfg["plot_kwargs_by_name"] = self.resolve_plot_kwargs_by_name(cfg=cfg)

    @staticmethod
    def resolve_registered_names(
        *,
        cfg: Mapping[str, Any],
        config_key: str,
        error_context: str,
    ) -> list[str]:
        raw = dict(cfg).get(config_key)
        if raw is None:
            return []
        if isinstance(raw, (list, tuple)):
            names = [str(value) for value in raw if str(value).strip()]
            return names
        raise ValueError(f"{error_context} must be a list of names when provided.")

    @staticmethod
    def resolve_default_plot_kwargs(*, cfg: Mapping[str, Any]) -> dict[str, Any]:
        raw = dict(cfg).get("default_plot_kwargs")
        if raw is None:
            return {}
        if not isinstance(raw, Mapping):
            raise ValueError("evaluation.default_plot_kwargs must be a mapping when provided.")
        return dict(raw)

    @staticmethod
    def resolve_plot_kwargs_by_name(*, cfg: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
        raw = dict(cfg).get("plot_kwargs_by_name")
        if raw is None:
            return {}
        if not isinstance(raw, Mapping):
            raise ValueError("evaluation.plot_kwargs_by_name must be a mapping when provided.")
        out: dict[str, dict[str, Any]] = {}
        for key, value in raw.items():
            if value is None:
                continue
            if not isinstance(value, Mapping):
                raise ValueError("evaluation.plot_kwargs_by_name values must be mappings.")
            out[str(key)] = dict(value)
        return out
