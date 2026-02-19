from zenml import step

from pioneerml.pipelines.training.positron_angle.dataset import PositronAngleDataset
from pioneerml.pipelines.training.positron_angle.services import PositronAngleTrainingService


@step
def train_positron_angle(
    dataset: PositronAngleDataset,
    pipeline_config: dict | None = None,
    hpo_params: dict | None = None,
):
    service = PositronAngleTrainingService(
        dataset=dataset,
        pipeline_config=pipeline_config,
        hpo_params=hpo_params,
    )
    return service.execute()
