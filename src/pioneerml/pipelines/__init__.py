"""ZenML pipelines for PIONEER ML (active pipelines only)."""

__all__: list[str] = []

try:
    from .training import training_pipeline

    __all__.extend(
        [
            "training_pipeline",
        ]
    )
except Exception:
    pass

try:
    from .inference import inference_pipeline

    __all__.extend(
        [
            "inference_pipeline",
        ]
    )
except Exception:
    pass
