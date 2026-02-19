from importlib import import_module

__all__ = ["group_splitting_inference_pipeline"]


def __getattr__(name: str):
    if name == "group_splitting_inference_pipeline":
        return import_module(
            "pioneerml.pipelines.inference.group_splitting.pipeline"
        ).group_splitting_inference_pipeline
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
