def resolve_step_config(config, key: str) -> dict | None:
    if config is None:
        return None
    if not isinstance(config, dict):
        raise TypeError(f"Expected mapping config, got {type(config).__name__}.")

    if key in config:
        raw = config.get(key)
        if raw is None:
            return None
        if isinstance(raw, dict):
            return dict(raw)
        raise TypeError(f"Expected dict for '{key}' config, got {type(raw).__name__}.")

    known = ("loader", "hpo", "train", "evaluate", "export")
    if any(step in config for step in known):
        return None
    return dict(config)
