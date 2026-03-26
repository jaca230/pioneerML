from __future__ import annotations

import json
import logging
from collections.abc import Mapping
from typing import Any

LOGGER = logging.getLogger(__name__)


def merge_nested_dicts(*, base: dict[str, Any], override: Mapping[str, Any] | None) -> dict[str, Any]:
    out = dict(base)
    if override is None:
        return out
    for key, value in dict(override).items():
        if isinstance(value, Mapping) and isinstance(out.get(key), dict):
            out[key] = merge_nested_dicts(base=dict(out[key]), override=value)
        else:
            out[key] = value
    return out


def log_loader_diagnostics(*, label: str, loader_provider: Any) -> dict[str, Any]:
    if loader_provider is None or not hasattr(loader_provider, "get_diagnostics_summary"):
        return {}
    try:
        summary = loader_provider.get_diagnostics_summary() or {}
    except Exception:
        return {}
    if summary:
        LOGGER.info("[loader_diagnostics][%s] %s", label, json.dumps(summary, sort_keys=True))
    return dict(summary)
