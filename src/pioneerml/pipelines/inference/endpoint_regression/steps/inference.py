from zenml import step

from pioneerml.pipelines.inference.endpoint_regression.services import (
    EndpointRegressorInferenceRunService,
)


@step(enable_cache=False)
def run_endpoint_regressor_inference(
    model_info: dict,
    inputs: dict,
    pipeline_config: dict | None = None,
) -> dict:
    service = EndpointRegressorInferenceRunService(pipeline_config=pipeline_config)
    return service.execute(model_info=model_info, inputs=inputs)
