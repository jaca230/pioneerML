from zenml import step

from pioneerml.pipelines.inference.group_classification.services import GroupClassifierInferenceModelLoaderService


@step(enable_cache=False)
def load_group_classifier_model(
    model_path: str | None = None,
    pipeline_config: dict | None = None,
) -> dict:
    service = GroupClassifierInferenceModelLoaderService(pipeline_config=pipeline_config)
    return service.execute(model_path=model_path)
