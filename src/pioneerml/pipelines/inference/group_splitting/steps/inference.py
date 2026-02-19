from zenml import step

from pioneerml.pipelines.inference.group_splitting.services import GroupSplitterInferenceRunService


@step(enable_cache=False)
def run_group_splitter_inference(
    model_info: dict,
    inputs: dict,
    pipeline_config: dict | None = None,
) -> dict:
    service = GroupSplitterInferenceRunService(pipeline_config=pipeline_config)
    return service.execute(model_info=model_info, inputs=inputs)
