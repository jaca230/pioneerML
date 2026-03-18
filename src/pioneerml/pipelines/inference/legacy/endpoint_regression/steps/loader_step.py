from zenml import step

from pioneerml.common.pipeline.steps import BaseLoaderFactoryInitStep


class EndpointRegressorInferenceInputsStep(BaseLoaderFactoryInitStep):
    step_key = "loader_factory_init"

    def default_config(self) -> dict:
        return {"loader_name": "endpoint_regression"}


@step(name="load_endpoint_regressor_inference_inputs", enable_cache=False)
def load_endpoint_regressor_inference_inputs_step(
    input_source_set: dict,
    pipeline_config: dict | None = None,
) -> dict:
    return EndpointRegressorInferenceInputsStep(pipeline_config=pipeline_config).execute(
        payloads={"input_source_set": input_source_set},
    )
