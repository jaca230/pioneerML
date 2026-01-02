"""ZenML inference pipelines."""

from pioneerml.zenml.pipelines.inference.upstream_inference_pipeline import upstream_inference_pipeline
from pioneerml.zenml.pipelines.inference.downstream_inference_pipeline import downstream_inference_pipeline

__all__ = ["upstream_inference_pipeline", "downstream_inference_pipeline"]
