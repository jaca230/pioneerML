from zenml import step

from pioneerml.pipelines.training.group_splitting.dataset import GroupSplitterDataset
from pioneerml.pipelines.training.group_splitting.services import GroupSplitterHPOService


@step
def tune_group_splitter(
    dataset: GroupSplitterDataset,
    pipeline_config: dict | None = None,
) -> dict:
    service = GroupSplitterHPOService(
        dataset=dataset,
        pipeline_config=pipeline_config,
    )
    return service.execute()
