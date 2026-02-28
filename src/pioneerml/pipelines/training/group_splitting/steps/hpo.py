from zenml import step

from pioneerml.common.loader import TrainingBatchBundle
from pioneerml.pipelines.training.group_splitting.services import GroupSplitterHPOService


@step
def tune_group_splitter(
    dataset: TrainingBatchBundle,
    pipeline_config: dict | None = None,
) -> dict:
    service = GroupSplitterHPOService(
        dataset=dataset,
        pipeline_config=pipeline_config,
    )
    return service.execute()
