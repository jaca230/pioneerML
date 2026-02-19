from zenml import step

from pioneerml.pipelines.training.group_classification.dataset import GroupClassifierDataset
from pioneerml.pipelines.training.group_classification.services import GroupClassifierEvaluationService


@step
def evaluate_group_classifier(
    module,
    dataset: GroupClassifierDataset,
    pipeline_config: dict | None = None,
) -> dict:
    service = GroupClassifierEvaluationService(
        dataset=dataset,
        module=module,
        pipeline_config=pipeline_config,
    )
    return service.execute()
