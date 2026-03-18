from __future__ import annotations

from zenml import pipeline

from pioneerml.common.pipeline.runners import InferencePipelineRunner

from .steps import (
    load_group_splitter_inference_inputs_step,
    load_group_splitter_writer_step,
    load_group_splitter_model_step,
    run_group_splitter_inference_step,
)

_RUNNER = InferencePipelineRunner(
    load_inputs_step=load_group_splitter_inference_inputs_step,
    load_writer_step=load_group_splitter_writer_step,
    load_model_step=load_group_splitter_model_step,
    run_inference_step=run_group_splitter_inference_step,
)


@pipeline
def group_splitting_inference_pipeline(
    input_source_set: dict,
    model_path: str | None = None,
    output_dir: str | None = None,
    output_path: str | None = None,
    pipeline_config: dict | None = None,
):
    return _RUNNER.run(
        loader_kwargs={"input_source_set": input_source_set},
        model_path=model_path,
        output_dir=output_dir,
        output_path=output_path,
        pipeline_config=pipeline_config,
    )
