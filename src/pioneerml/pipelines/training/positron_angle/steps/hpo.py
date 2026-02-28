from zenml import step

from pioneerml.common.loader import TrainingBatchBundle
from pioneerml.pipelines.training.positron_angle.services import PositronAngleHPOService


@step
def tune_positron_angle(
    dataset: TrainingBatchBundle,
    pipeline_config: dict | None = None,
) -> dict:
    service = PositronAngleHPOService(
        dataset=dataset,
        pipeline_config=pipeline_config,
    )
    return service.execute()
