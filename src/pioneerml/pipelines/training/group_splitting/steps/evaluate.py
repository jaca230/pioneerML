from zenml import step

from pioneerml.pipelines.training.group_splitting.dataset import GroupSplitterDataset
from pioneerml.pipelines.training.group_splitting.services import GroupSplitterEvaluationService


@step
def evaluate_group_splitter(
    module,
    dataset: GroupSplitterDataset,
    pipeline_config: dict | None = None,
) -> dict:
    service = GroupSplitterEvaluationService(
        dataset=dataset,
        module=module,
        pipeline_config=pipeline_config,
    )
    return service.execute()
