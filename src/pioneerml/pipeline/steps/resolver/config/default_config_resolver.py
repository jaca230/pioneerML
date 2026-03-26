from __future__ import annotations

from typing import Any

from .base_config_resolver import BaseConfigResolver


def _deep_fill_defaults(*, cfg: dict[str, Any], defaults: dict[str, Any]) -> None:
    for key, default_value in defaults.items():
        if key not in cfg:
            cfg[key] = default_value
            continue
        current = cfg[key]
        if isinstance(current, dict) and isinstance(default_value, dict):
            _deep_fill_defaults(cfg=current, defaults=default_value)


def _deep_merge_override(*, base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    out = dict(base)
    for key, value in dict(override).items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _deep_merge_override(base=dict(out[key]), override=value)
        else:
            out[key] = value
    return out


class DefaultConfigResolver(BaseConfigResolver):
    """Fill step config with step defaults for keys not set by user."""

    def resolve(self, *, cfg: dict[str, Any]) -> None:
        defaults = dict(getattr(self.step, "DEFAULT_CONFIG", {}) or {})
        default_fn = getattr(self.step, "default_config", None)
        if callable(default_fn):
            maybe = default_fn()
            if isinstance(maybe, dict):
                defaults = _deep_merge_override(base=defaults, override=maybe)
        _deep_fill_defaults(cfg=cfg, defaults=defaults)
