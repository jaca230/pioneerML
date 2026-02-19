"""
ZenML pipelines for PIONEER ML.
"""

from importlib import import_module

__all__ = [
    "group_classification_pipeline",
    "group_splitting_pipeline",
    "endpoint_regression_pipeline",
    "event_splitting_pipeline",
    "pion_stop_regression_pipeline",
    "positron_angle_regression_pipeline",
    "group_classification_inference_pipeline",
    "group_splitting_inference_pipeline",
    "endpoint_regression_inference_pipeline",
    "event_splitting_inference_pipeline",
    "pion_stop_regression_inference_pipeline",
    "positron_angle_regression_inference_pipeline",
]


def __getattr__(name: str):
    if name == "group_classification_pipeline":
        return import_module("pioneerml.pipelines.training.group_classification.pipeline").group_classification_pipeline
    if name == "group_splitting_pipeline":
        return import_module("pioneerml.pipelines.training.group_splitting.pipeline").group_splitting_pipeline
    if name == "endpoint_regression_pipeline":
        return import_module("pioneerml.pipelines.training.endpoint_regression.pipeline").endpoint_regression_pipeline
    if name == "event_splitting_pipeline":
        return import_module("pioneerml.pipelines.training.event_splitting.pipeline").event_splitting_pipeline
    if name == "pion_stop_regression_pipeline":
        return import_module("pioneerml.pipelines.training.pion_stop.pipeline").pion_stop_regression_pipeline
    if name == "positron_angle_regression_pipeline":
        return import_module("pioneerml.pipelines.training.positron_angle.pipeline").positron_angle_regression_pipeline
    if name == "group_classification_inference_pipeline":
        return import_module(
            "pioneerml.pipelines.inference.group_classification.pipeline"
        ).group_classification_inference_pipeline
    if name == "group_splitting_inference_pipeline":
        return import_module("pioneerml.pipelines.inference.group_splitting.pipeline").group_splitting_inference_pipeline
    if name == "endpoint_regression_inference_pipeline":
        return import_module(
            "pioneerml.pipelines.inference.endpoint_regression.pipeline"
        ).endpoint_regression_inference_pipeline
    if name == "event_splitting_inference_pipeline":
        return import_module(
            "pioneerml.pipelines.inference.event_splitting.pipeline"
        ).event_splitting_inference_pipeline
    if name == "pion_stop_regression_inference_pipeline":
        return import_module("pioneerml.pipelines.inference.pion_stop.pipeline").pion_stop_regression_inference_pipeline
    if name == "positron_angle_regression_inference_pipeline":
        return import_module(
            "pioneerml.pipelines.inference.positron_angle.pipeline"
        ).positron_angle_regression_inference_pipeline
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
