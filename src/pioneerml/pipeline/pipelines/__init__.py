"""Unified config-driven pipelines."""

from .inference import inference_pipeline
from .training import training_pipeline

__all__ = ["training_pipeline", "inference_pipeline"]
