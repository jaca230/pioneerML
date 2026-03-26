from __future__ import annotations

from collections.abc import Mapping
import inspect
from typing import Any

from .base import BasePluginBuilder


class DefaultPluginBuilder(BasePluginBuilder):
    """Default plugin materialization strategy.

    Resolution order:
    1) plugin.from_plugin(config=..., namespace=..., name=...)
    2) plugin.from_factory(config=..., namespace=..., name=...)
    3) plugin class constructor using config kwargs / flexible kwargs
    4) callable plugin(config=..., namespace=..., name=...)
    5) return plugin value directly
    """

    def build(
        self,
        *,
        plugin: Any,
        namespace: str,
        name: str,
        config: Mapping[str, Any] | None = None,
    ) -> Any:
        cfg = dict(config or {})
        call_ctx = {
            "config": cfg,
            "namespace": str(namespace),
            "name": str(name),
        }

        from_plugin = getattr(plugin, "from_plugin", None)
        if callable(from_plugin):
            return self._invoke_with_supported_kwargs(from_plugin, {**call_ctx, **cfg})

        from_factory = getattr(plugin, "from_factory", None)
        if callable(from_factory):
            return self._invoke_with_supported_kwargs(from_factory, {**call_ctx, **cfg})

        if inspect.isclass(plugin):
            if cfg:
                try:
                    return plugin(**cfg)
                except TypeError:
                    pass
            return self._invoke_with_supported_kwargs(plugin, {**call_ctx, **cfg})

        if callable(plugin):
            return self._invoke_with_supported_kwargs(plugin, {**call_ctx, **cfg})

        return plugin

    @staticmethod
    def _invoke_with_supported_kwargs(fn: Any, kwargs: Mapping[str, Any]) -> Any:
        sig = inspect.signature(fn)
        params = sig.parameters
        has_var_kw = any(param.kind is inspect.Parameter.VAR_KEYWORD for param in params.values())
        call_kwargs: dict[str, Any] = {}
        for key, value in dict(kwargs).items():
            if has_var_kw or key in params:
                call_kwargs[key] = value
        return fn(**call_kwargs)
