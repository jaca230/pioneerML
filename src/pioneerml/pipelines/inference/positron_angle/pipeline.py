from zenml import pipeline

from .steps.inference import run_positron_angle_inference
from .steps.loader import load_positron_angle_inference_inputs
from .steps.model_loader import load_positron_angle_model
from .steps.save_predictions import save_positron_angle_predictions


@pipeline
def positron_angle_regression_inference_pipeline(
    parquet_paths: list[str],
    group_probs_parquet_paths: list[str] | None = None,
    group_splitter_parquet_paths: list[str] | None = None,
    endpoint_parquet_paths: list[str] | None = None,
    event_splitter_parquet_paths: list[str] | None = None,
    pion_stop_parquet_paths: list[str] | None = None,
    model_path: str | None = None,
    output_dir: str | None = None,
    output_path: str | None = None,
    metrics_path: str | None = None,
    pipeline_config: dict | None = None,
):
    inputs = load_positron_angle_inference_inputs(
        parquet_paths=parquet_paths,
        group_probs_parquet_paths=group_probs_parquet_paths,
        group_splitter_parquet_paths=group_splitter_parquet_paths,
        endpoint_parquet_paths=endpoint_parquet_paths,
        event_splitter_parquet_paths=event_splitter_parquet_paths,
        pion_stop_parquet_paths=pion_stop_parquet_paths,
        pipeline_config=pipeline_config,
    )
    model_info = load_positron_angle_model(model_path=model_path, pipeline_config=pipeline_config)
    outputs = run_positron_angle_inference(model_info=model_info, inputs=inputs, pipeline_config=pipeline_config)
    return save_positron_angle_predictions(
        inference_outputs=outputs,
        output_dir=output_dir,
        output_path=output_path,
        metrics_path=metrics_path,
        pipeline_config=pipeline_config,
    )
