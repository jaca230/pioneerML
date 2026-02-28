from zenml import step

from pioneerml.common.loader import TrainingBatchBundle
from pioneerml.pipelines.training.group_splitting.services import GroupSplitterEvaluationService


@step
def evaluate_group_splitter(
    module,
    dataset: TrainingBatchBundle,
    pipeline_config: dict | None = None,
) -> dict:
    service = GroupSplitterEvaluationService(
        dataset=dataset,
        module=module,
        pipeline_config=pipeline_config,
    )
    return service.execute()
