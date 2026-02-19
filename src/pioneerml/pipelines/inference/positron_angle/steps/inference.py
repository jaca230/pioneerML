from zenml import step

from pioneerml.pipelines.inference.positron_angle.services import (
    PositronAngleInferenceRunService,
)


@step(enable_cache=False)
def run_positron_angle_inference(
    model_info: dict,
    inputs: dict,
    pipeline_config: dict | None = None,
) -> dict:
    service = PositronAngleInferenceRunService(pipeline_config=pipeline_config)
    return service.execute(model_info=model_info, inputs=inputs)
