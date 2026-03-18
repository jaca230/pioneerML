from __future__ import annotations

from collections.abc import Callable, Mapping
import inspect
from typing import Any


def _invoke_with_supported_kwargs(fn: Callable[..., Any], kwargs: Mapping[str, Any]) -> Any:
    sig = inspect.signature(fn)
    params = sig.parameters
    has_var_kw = any(param.kind is inspect.Parameter.VAR_KEYWORD for param in params.values())
    call_kwargs: dict[str, Any] = {}
    for key, value in dict(kwargs).items():
        if has_var_kw or key in params:
            call_kwargs[key] = value
    return fn(**call_kwargs)


class DefaultPluginBuilder:
    """Default plugin materialization strategy.

    Resolution order:
    1) plugin.from_plugin(config=..., context=..., namespace=..., name=...)
    2) plugin.from_factory(config=..., context=..., namespace=..., name=...)
    3) plugin class constructor using config kwargs / flexible kwargs
    4) callable plugin(config=..., context=..., namespace=..., name=...)
    5) return plugin value directly
    """

    def build(
        self,
        *,
        plugin: Any,
        namespace: str,
        name: str,
        config: Mapping[str, Any] | None = None,
        context: Mapping[str, Any] | None = None,
    ) -> Any:
        cfg = dict(config or {})
        ctx = dict(context or {})
        call_ctx = {
            "config": cfg,
            "context": ctx,
            "namespace": str(namespace),
            "name": str(name),
        }

        from_plugin = getattr(plugin, "from_plugin", None)
        if callable(from_plugin):
            return _invoke_with_supported_kwargs(from_plugin, call_ctx)

        from_factory = getattr(plugin, "from_factory", None)
        if callable(from_factory):
            return _invoke_with_supported_kwargs(from_factory, call_ctx)

        if inspect.isclass(plugin):
            if cfg:
                try:
                    return plugin(**cfg)
                except TypeError:
                    # Fall through to more permissive constructor invocation.
                    pass
            return _invoke_with_supported_kwargs(plugin, call_ctx)

        if callable(plugin):
            return _invoke_with_supported_kwargs(plugin, call_ctx)

        return plugin
