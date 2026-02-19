from importlib import import_module

__all__ = ["pion_stop_regression_pipeline"]


def __getattr__(name: str):
    if name == "pion_stop_regression_pipeline":
        return import_module("pioneerml.pipelines.training.pion_stop.pipeline").pion_stop_regression_pipeline
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
