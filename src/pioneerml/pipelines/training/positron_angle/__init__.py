from importlib import import_module

__all__ = ["positron_angle_regression_pipeline"]


def __getattr__(name: str):
    if name == "positron_angle_regression_pipeline":
        return import_module("pioneerml.pipelines.training.positron_angle.pipeline").positron_angle_regression_pipeline
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
