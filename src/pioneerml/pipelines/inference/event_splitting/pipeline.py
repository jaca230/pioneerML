from zenml import pipeline

from .steps.inference import run_event_splitter_inference
from .steps.loader import load_event_splitter_inference_inputs
from .steps.model_loader import load_event_splitter_model
from .steps.save_predictions import save_event_splitter_predictions


@pipeline
def event_splitting_inference_pipeline(
    parquet_paths: list[str],
    group_probs_parquet_paths: list[str] | None = None,
    group_splitter_parquet_paths: list[str] | None = None,
    endpoint_parquet_paths: list[str] | None = None,
    model_path: str | None = None,
    output_dir: str | None = None,
    output_path: str | None = None,
    metrics_path: str | None = None,
    pipeline_config: dict | None = None,
):
    inputs = load_event_splitter_inference_inputs(
        parquet_paths=parquet_paths,
        group_probs_parquet_paths=group_probs_parquet_paths,
        group_splitter_parquet_paths=group_splitter_parquet_paths,
        endpoint_parquet_paths=endpoint_parquet_paths,
        pipeline_config=pipeline_config,
    )
    model_info = load_event_splitter_model(model_path=model_path, pipeline_config=pipeline_config)
    outputs = run_event_splitter_inference(model_info=model_info, inputs=inputs, pipeline_config=pipeline_config)
    return save_event_splitter_predictions(
        inference_outputs=outputs,
        output_dir=output_dir,
        output_path=output_path,
        metrics_path=metrics_path,
        pipeline_config=pipeline_config,
    )
