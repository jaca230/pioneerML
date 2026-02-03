from __future__ import annotations

from typing import Mapping


def resolve_step_config(pipeline_config: dict | None, step_name: str) -> dict | None:
    if pipeline_config is None:
        return None
    if step_name in pipeline_config:
        cfg = pipeline_config.get(step_name)
        return cfg if isinstance(cfg, dict) else None
    if isinstance(pipeline_config, Mapping):
        return dict(pipeline_config)
    return None
