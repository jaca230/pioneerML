from zenml import step

from pioneerml.common.pipeline.steps import BaseModelHandleBuilderStep


class GroupClassifierInferenceModelLoaderStep(BaseModelHandleBuilderStep):
    step_key = "model_handle_builder"

    def default_config(self) -> dict:
        return {"model_subdir": "groupclassifier"}


@step(name="load_group_classifier_model", enable_cache=False)
def load_group_classifier_model_step(
    model_path: str | None = None,
    pipeline_config: dict | None = None,
) -> dict:
    payloads = {"model_path": model_path} if model_path is not None else None
    return GroupClassifierInferenceModelLoaderStep(pipeline_config=pipeline_config).execute(payloads=payloads)
