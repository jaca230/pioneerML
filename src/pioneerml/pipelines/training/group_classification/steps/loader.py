from zenml import step

from pioneerml.common.loader import GroupClassifierGraphLoader
from pioneerml.common.zenml.materializers import GroupClassifierDatasetMaterializer
from pioneerml.pipelines.training.group_classification.dataset import GroupClassifierDataset
from pioneerml.pipelines.training.group_classification.steps.config import resolve_step_config


@step(enable_cache=False, output_materializers=GroupClassifierDatasetMaterializer)
def load_group_classifier_dataset(
    parquet_paths: list[str],
    pipeline_config: dict | None = None,
) -> GroupClassifierDataset:
    step_config = resolve_step_config(pipeline_config, "loader") or {}
    config_json = dict(step_config.get("config_json") or {})

    loader = GroupClassifierGraphLoader(
        parquet_paths=[str(p) for p in parquet_paths],
        mode=str(config_json.get("mode", "train")),
        batch_size=max(1, int(config_json.get("batch_size", 64))),
        row_groups_per_chunk=max(1, int(config_json.get("chunk_row_groups", config_json.get("row_groups_per_chunk", 4)))),
        num_workers=max(0, int(config_json.get("chunk_workers", config_json.get("num_workers", 0)))),
    )

    data, targets = loader.empty_data()
    data.source_parquet_paths = list(loader.parquet_paths)
    return GroupClassifierDataset(data=data, targets=targets, loader=loader)
