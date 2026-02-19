from importlib import import_module

__all__ = ["event_splitting_pipeline"]


def __getattr__(name: str):
    if name == "event_splitting_pipeline":
        return import_module("pioneerml.pipelines.training.event_splitting.pipeline").event_splitting_pipeline
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
