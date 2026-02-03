from typing import Mapping


def resolve_step_config(config: Mapping | None, key: str) -> dict | None:
    if config is None:
        return None
    if not isinstance(config, Mapping):
        raise TypeError(f"Expected mapping config, got {type(config).__name__}.")
    if key in config:
        raw = config.get(key)
        if raw is None:
            return None
        if isinstance(raw, Mapping):
            return dict(raw)
        raise TypeError(f"Expected dict for '{key}' config, got {type(raw).__name__}.")
    if any(k in config for k in ("loader", "hpo", "train", "evaluate", "export")):
        return None
    return dict(config)
