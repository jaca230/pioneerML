from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import optuna

from .base_search_space import BaseSearchSpace
from .factory.registry import REGISTRY as SEARCH_SPACE_REGISTRY
from .parameters import SearchParameterFactory


@SEARCH_SPACE_REGISTRY.register("config")
class ConfigSearchSpace(BaseSearchSpace):
    def __init__(
        self,
        *,
        search_space: Mapping[str, Any] | None = None,
    ) -> None:
        self.search_space = dict(search_space or {})

    def default_search_space(self) -> dict[str, Any]:
        return dict(self.search_space)

    def suggest(self, *, trial: optuna.Trial, search_space: Mapping[str, Any] | None = None) -> dict[str, Any]:
        merged = dict(self.default_search_space())
        if isinstance(search_space, Mapping):
            merged.update(dict(search_space))

        out: dict[str, Any] = {}
        for name, raw_spec in merged.items():
            key = str(name)
            param = self._build_parameter(raw_spec=raw_spec)
            out[key] = param.suggest(trial=trial, name=key)
        return out

    @staticmethod
    def _infer_parameter_type(*, spec: Mapping[str, Any]) -> str:
        explicit = str(spec.get("type", "auto")).strip().lower()
        if explicit in {"exponent_int", "exp2_int", "exp2", "power_of_two", "pow2"}:
            return "exponent_int"
        if "min_exp" in spec and "max_exp" in spec:
            return "exponent_int"
        if "value" in spec:
            return "fixed"
        if "choices" in spec:
            return "categorical"
        if "low" in spec and "high" in spec:
            if explicit in {"int", "float"}:
                return explicit
            low, high = spec.get("low"), spec.get("high")
            if isinstance(low, int) and isinstance(high, int):
                return "int"
            return "float"
        raise ValueError(
            "Search-space parameter must provide one of: value, choices, low/high, or min_exp/max_exp."
        )

    def _build_parameter(self, *, raw_spec: Any):
        if isinstance(raw_spec, Mapping):
            spec = dict(raw_spec)
            param_type = self._infer_parameter_type(spec=spec)
            if "type" in spec:
                spec.pop("type", None)
            return SearchParameterFactory(search_parameter_name=param_type).build(config=spec)
        return SearchParameterFactory(search_parameter_name="fixed").build(config={"value": raw_spec})
