"""Current training pipelines."""

from .endpoint_regression import endpoint_regression_pipeline
from .group_classification import group_classification_pipeline
from .group_splitting import group_splitting_pipeline

__all__ = [
    "group_classification_pipeline",
    "group_splitting_pipeline",
    "endpoint_regression_pipeline",
]
