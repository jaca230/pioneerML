from zenml import step

from pioneerml.pipelines.inference.pion_stop.services import (
    PionStopInferenceRunService,
)


@step(enable_cache=False)
def run_pion_stop_inference(
    model_info: dict,
    inputs: dict,
    pipeline_config: dict | None = None,
) -> dict:
    service = PionStopInferenceRunService(pipeline_config=pipeline_config)
    return service.execute(model_info=model_info, inputs=inputs)
