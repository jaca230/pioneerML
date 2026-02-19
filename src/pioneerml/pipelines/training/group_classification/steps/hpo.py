from zenml import step

from pioneerml.pipelines.training.group_classification.dataset import GroupClassifierDataset
from pioneerml.pipelines.training.group_classification.services import GroupClassifierHPOService


@step
def tune_group_classifier(
    dataset: GroupClassifierDataset,
    pipeline_config: dict | None = None,
) -> dict:
    service = GroupClassifierHPOService(
        dataset=dataset,
        pipeline_config=pipeline_config,
    )
    return service.execute()
