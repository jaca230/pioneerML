from zenml import step

from pioneerml.pipelines.inference.positron_angle.services import (
    PositronAngleSavePredictionsService,
)


@step(enable_cache=False)
def save_positron_angle_predictions(
    inference_outputs: dict,
    output_dir: str | None = None,
    output_path: str | None = None,
    metrics_path: str | None = None,
    pipeline_config: dict | None = None,
) -> dict:
    service = PositronAngleSavePredictionsService(pipeline_config=pipeline_config)
    return service.execute(
        inference_outputs=inference_outputs,
        output_dir=output_dir,
        output_path=output_path,
        metrics_path=metrics_path,
    )
