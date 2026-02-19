from zenml import step

from pioneerml.pipelines.inference.group_classification.services import GroupClassifierInferenceRunService


@step(enable_cache=False)
def run_group_classifier_inference(
    model_info: dict,
    inputs: dict,
    pipeline_config: dict | None = None,
) -> dict:
    service = GroupClassifierInferenceRunService(pipeline_config=pipeline_config)
    return service.execute(model_info=model_info, inputs=inputs)
