from importlib import import_module

__all__ = ["positron_angle_regression_inference_pipeline"]


def __getattr__(name: str):
    if name == "positron_angle_regression_inference_pipeline":
        return import_module(
            "pioneerml.pipelines.inference.positron_angle.pipeline"
        ).positron_angle_regression_inference_pipeline
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
