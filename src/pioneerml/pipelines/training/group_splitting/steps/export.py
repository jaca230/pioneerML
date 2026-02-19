from zenml import step

from pioneerml.pipelines.training.group_splitting.dataset import GroupSplitterDataset
from pioneerml.pipelines.training.group_splitting.services import GroupSplitterExportService


@step
def export_group_splitter(
    module,
    dataset: GroupSplitterDataset,
    pipeline_config: dict | None = None,
    hpo_params: dict | None = None,
    metrics: dict | None = None,
) -> dict:
    service = GroupSplitterExportService(
        dataset=dataset,
        module=module,
        pipeline_config=pipeline_config,
        hpo_params=hpo_params,
        metrics=metrics,
    )
    return service.execute()
