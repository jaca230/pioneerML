from zenml import step

from pioneerml.pipelines.training.group_splitting.dataset import GroupSplitterDataset
from pioneerml.pipelines.training.group_splitting.services import GroupSplitterTrainingService


@step
def train_group_splitter(
    dataset: GroupSplitterDataset,
    pipeline_config: dict | None = None,
    hpo_params: dict | None = None,
):
    service = GroupSplitterTrainingService(
        dataset=dataset,
        pipeline_config=pipeline_config,
        hpo_params=hpo_params,
    )
    return service.execute()
