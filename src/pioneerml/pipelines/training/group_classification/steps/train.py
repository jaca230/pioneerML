from zenml import step

from pioneerml.pipelines.training.group_classification.dataset import GroupClassifierDataset
from pioneerml.pipelines.training.group_classification.services import GroupClassifierTrainingService


@step
def train_group_classifier(
    dataset: GroupClassifierDataset,
    pipeline_config: dict | None = None,
    hpo_params: dict | None = None,
):
    service = GroupClassifierTrainingService(
        dataset=dataset,
        pipeline_config=pipeline_config,
        hpo_params=hpo_params,
    )
    return service.execute()
