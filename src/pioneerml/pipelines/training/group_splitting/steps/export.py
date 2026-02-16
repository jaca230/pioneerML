from zenml import step

from pioneerml.common.loader import GroupSplitterGraphLoader
from pioneerml.common.pipeline_utils.export import TorchscriptExporter
from pioneerml.pipelines.training.group_splitting.dataset import GroupSplitterDataset
from pioneerml.pipelines.training.group_splitting.steps.config import resolve_step_config


_EXPORTER = TorchscriptExporter()


def _build_export_example(dataset: GroupSplitterDataset):
    data = dataset.data
    if hasattr(data, "batch"):
        return (data.x, data.edge_index, data.edge_attr, data.batch, data.group_total_energy, data.group_probs)
    loader_obj = getattr(dataset, "loader", None)
    if isinstance(loader_obj, GroupSplitterGraphLoader):
        loader = loader_obj.with_runtime(batch_size=1, row_groups_per_chunk=1, num_workers=0).make_dataloader(
            shuffle_batches=False
        )
        for batch in loader:
            return (batch.x, batch.edge_index, batch.edge_attr, batch.batch, batch.group_total_energy, batch.group_probs)
    return None


@step
def export_group_splitter(
    module,
    dataset: GroupSplitterDataset,
    pipeline_config: dict | None = None,
    hpo_params: dict | None = None,
    metrics: dict | None = None,
) -> dict:
    cfg = resolve_step_config(pipeline_config, "export") or {}
    return _EXPORTER.export(
        module=module,
        dataset=dataset,
        cfg=cfg,
        pipeline_config=pipeline_config,
        hpo_params=hpo_params,
        metrics=metrics,
        default_export_dir="trained_models/groupsplitter",
        default_prefix="groupsplitter",
        example_builder=_build_export_example,
    )
