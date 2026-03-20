from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping
import inspect
from typing import Any

import optuna


class BaseSearchSpace(ABC):
    @abstractmethod
    def default_search_space(self) -> dict[str, Any]:
        raise NotImplementedError

    @abstractmethod
    def suggest(self, *, trial: optuna.Trial, search_space: Mapping[str, Any] | None = None) -> dict[str, Any]:
        raise NotImplementedError

    @staticmethod
    def constructor_param_names(cls_or_callable: Any) -> set[str]:
        target = cls_or_callable.__init__ if inspect.isclass(cls_or_callable) else cls_or_callable
        sig = inspect.signature(target)
        names: set[str] = set()
        for name, param in sig.parameters.items():
            if name == "self":
                continue
            if param.kind in (inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.KEYWORD_ONLY):
                names.add(str(name))
        return names

    @classmethod
    def partition_suggested_params(
        cls,
        *,
        suggested: Mapping[str, Any],
        model_cls: Any,
        module_cls: Any,
    ) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
        model_keys = cls.constructor_param_names(model_cls)
        module_keys = cls.constructor_param_names(module_cls)

        model_updates: dict[str, Any] = {}
        module_updates: dict[str, Any] = {}
        runtime_updates: dict[str, Any] = {}

        for key, value in dict(suggested).items():
            in_model = key in model_keys
            in_module = key in module_keys
            if in_model and in_module:
                raise ValueError(
                    f"Suggested parameter '{key}' exists in both model and module constructors. "
                    "Rename it or map it explicitly in configuration."
                )
            if in_model:
                model_updates[str(key)] = value
                continue
            if in_module:
                module_updates[str(key)] = value
                continue
            runtime_updates[str(key)] = value

        return model_updates, module_updates, runtime_updates
