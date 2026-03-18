"""ZenML pipelines for PIONEER ML (active pipelines only)."""

__all__: list[str] = []

try:
    from .training import endpoint_regression_pipeline, group_classification_pipeline, group_splitting_pipeline

    __all__.extend(
        [
            "endpoint_regression_pipeline",
            "group_classification_pipeline",
            "group_splitting_pipeline",
        ]
    )
except Exception:
    pass

try:
    from .inference import (
        endpoint_regression_inference_pipeline,
        group_classification_inference_pipeline,
        group_splitting_inference_pipeline,
    )

    __all__.extend(
        [
            "endpoint_regression_inference_pipeline",
            "group_classification_inference_pipeline",
            "group_splitting_inference_pipeline",
        ]
    )
except Exception:
    pass
