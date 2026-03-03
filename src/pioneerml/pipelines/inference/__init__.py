"""Current inference pipelines."""

from .group_classification import group_classification_inference_pipeline
from .group_splitting import group_splitting_inference_pipeline
from .endpoint_regression import endpoint_regression_inference_pipeline

__all__ = [
    "group_classification_inference_pipeline",
    "group_splitting_inference_pipeline",
    "endpoint_regression_inference_pipeline",
]
