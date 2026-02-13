from zenml import pipeline

from .steps.export import export_group_classifier_predictions
from .steps.inference import run_group_classifier_inference
from .steps.loader import load_group_classifier_inference_inputs
from .steps.model_loader import load_group_classifier_model


@pipeline
def group_classification_inference_pipeline(
    parquet_paths: list[str],
    model_path: str | None = None,
    output_dir: str | None = None,
    output_path: str | None = None,
    metrics_path: str | None = None,
    pipeline_config: dict | None = None,
):
    inputs = load_group_classifier_inference_inputs(
        parquet_paths=parquet_paths,
        pipeline_config=pipeline_config,
    )
    model_info = load_group_classifier_model(model_path=model_path, pipeline_config=pipeline_config)
    outputs = run_group_classifier_inference(model_info=model_info, inputs=inputs, pipeline_config=pipeline_config)
    return export_group_classifier_predictions(
        inference_outputs=outputs,
        output_dir=output_dir,
        output_path=output_path,
        metrics_path=metrics_path,
        pipeline_config=pipeline_config,
    )
