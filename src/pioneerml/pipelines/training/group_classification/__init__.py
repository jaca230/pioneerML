from importlib import import_module

__all__ = ["group_classification_pipeline"]


def __getattr__(name: str):
    if name == "group_classification_pipeline":
        return import_module("pioneerml.pipelines.training.group_classification.pipeline").group_classification_pipeline
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
