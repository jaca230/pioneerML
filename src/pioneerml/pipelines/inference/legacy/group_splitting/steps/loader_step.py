from zenml import step

from pioneerml.common.pipeline.steps import BaseLoaderFactoryInitStep


class GroupSplitterInferenceInputsStep(BaseLoaderFactoryInitStep):
    step_key = "loader_factory_init"

    def default_config(self) -> dict:
        return {"loader_name": "group_splitter"}


@step(name="load_group_splitter_inference_inputs", enable_cache=False)
def load_group_splitter_inference_inputs_step(
    input_source_set: dict,
    pipeline_config: dict | None = None,
) -> dict:
    return GroupSplitterInferenceInputsStep(pipeline_config=pipeline_config).execute(
        payloads={"input_source_set": input_source_set},
    )
