from zenml import step

from pioneerml.common.loader import TrainingBatchBundle
from pioneerml.pipelines.training.group_classification.services import GroupClassifierHPOService


@step
def tune_group_classifier(
    dataset: TrainingBatchBundle,
    pipeline_config: dict | None = None,
) -> dict:
    service = GroupClassifierHPOService(
        dataset=dataset,
        pipeline_config=pipeline_config,
    )
    return service.execute()
