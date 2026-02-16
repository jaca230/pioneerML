from zenml import step

from pioneerml.common.pipeline_utils.export import TorchscriptExporter
from pioneerml.pipelines.training.group_classification.dataset import GroupClassifierDataset
from pioneerml.pipelines.training.group_classification.steps.config import resolve_step_config
from .train import _build_stage_loader, _resolve_stage_loader_config

_EXPORTER = TorchscriptExporter()


def _build_export_example(dataset: GroupClassifierDataset):
    data = dataset.data
    if hasattr(data, "batch"):
        return (data.x, data.edge_index, data.edge_attr, data.batch)
    loader_obj = getattr(dataset, "loader", None)
    if loader_obj is not None:
        loader = loader_obj.with_runtime(batch_size=1, row_groups_per_chunk=1, num_workers=0).make_dataloader(
            shuffle_batches=False
        )
        for batch in loader:
            return (batch.x, batch.edge_index, batch.edge_attr, batch.batch)
    return None


@step
def export_group_classifier(
    module,
    parquet_paths: list[str],
    pipeline_config: dict | None = None,
    hpo_params: dict | None = None,
    metrics: dict | None = None,
) -> dict:
    cfg = resolve_step_config(pipeline_config, "export") or {}
    export_loader_cfg = {
        "batch_size": 1,
        "chunk_row_groups": 1,
        "chunk_workers": 0,
    }
    if isinstance(cfg.get("loader_config"), dict):
        export_loader_cfg["loader_config"] = cfg["loader_config"]
    loader_cfg = _resolve_stage_loader_config(export_loader_cfg, stage="val", forced_batch_size=1)
    loader = _build_stage_loader(parquet_paths=[str(p) for p in parquet_paths], loader_cfg=loader_cfg)
    data, targets = loader.empty_data()
    data.source_parquet_paths = list(loader.parquet_paths)
    dataset = GroupClassifierDataset(data=data, targets=targets, loader=loader)
    return _EXPORTER.export(
        module=module,
        dataset=dataset,
        cfg=cfg,
        pipeline_config=pipeline_config,
        hpo_params=hpo_params,
        metrics=metrics,
        default_export_dir="trained_models/groupclassifier",
        default_prefix="groupclassifier",
        example_builder=_build_export_example,
    )
