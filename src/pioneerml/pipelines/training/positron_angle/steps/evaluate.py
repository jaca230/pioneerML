from zenml import step

from pioneerml.pipelines.training.positron_angle.dataset import PositronAngleDataset
from pioneerml.pipelines.training.positron_angle.services import PositronAngleEvaluationService


@step
def evaluate_positron_angle(
    module,
    dataset: PositronAngleDataset,
    pipeline_config: dict | None = None,
) -> dict:
    service = PositronAngleEvaluationService(
        dataset=dataset,
        module=module,
        pipeline_config=pipeline_config,
    )
    return service.execute()
