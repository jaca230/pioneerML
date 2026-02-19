from importlib import import_module

__all__ = ["endpoint_regression_pipeline"]


def __getattr__(name: str):
    if name == "endpoint_regression_pipeline":
        return import_module("pioneerml.pipelines.training.endpoint_regression.pipeline").endpoint_regression_pipeline
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
