from zenml import step

from pioneerml.pipelines.training.positron_angle.dataset import PositronAngleDataset
from pioneerml.pipelines.training.positron_angle.services import PositronAngleHPOService


@step
def tune_positron_angle(
    dataset: PositronAngleDataset,
    pipeline_config: dict | None = None,
) -> dict:
    service = PositronAngleHPOService(
        dataset=dataset,
        pipeline_config=pipeline_config,
    )
    return service.execute()
