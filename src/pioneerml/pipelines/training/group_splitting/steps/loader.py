from zenml import step

from pioneerml.common.loader import GroupSplitterGraphLoader
from pioneerml.common.zenml.materializers import GroupSplitterDatasetMaterializer
from pioneerml.pipelines.training.group_splitting.dataset import GroupSplitterDataset


@step(enable_cache=False, output_materializers=GroupSplitterDatasetMaterializer)
def load_group_splitter_dataset(
    parquet_paths: list[str],
    group_probs_parquet_paths: list[str] | None = None,
    pipeline_config: dict | None = None,
) -> GroupSplitterDataset:
    config_json = {}
    if isinstance(pipeline_config, dict):
        loader_cfg = pipeline_config.get("loader")
        if isinstance(loader_cfg, dict):
            config_json = dict(loader_cfg.get("config_json") or {})

    loader = GroupSplitterGraphLoader(
        parquet_paths=[str(p) for p in parquet_paths],
        group_probs_parquet_paths=[str(p) for p in group_probs_parquet_paths]
        if group_probs_parquet_paths is not None
        else None,
        mode=str(config_json.get("mode", "train")),
        use_group_probs=bool(config_json.get("use_group_probs", True)),
        batch_size=max(1, int(config_json.get("batch_size", 64))),
        row_groups_per_chunk=max(1, int(config_json.get("chunk_row_groups", config_json.get("row_groups_per_chunk", 4)))),
        num_workers=max(0, int(config_json.get("chunk_workers", config_json.get("num_workers", 0)))),
        split=(None if config_json.get("split") in (None, "", "none") else str(config_json.get("split"))),
        train_fraction=float(config_json.get("train_fraction", 0.9)),
        val_fraction=float(config_json.get("val_fraction", 0.05)),
        test_fraction=float(config_json.get("test_fraction", 0.05)),
        split_seed=int(config_json.get("split_seed", 0)),
        sample_fraction=(
            None
            if config_json.get("sample_fraction") in (None, "", "none")
            else float(config_json.get("sample_fraction"))
        ),
    )
    data, targets = loader.empty_data()
    data.source_parquet_paths = list(loader.parquet_paths)
    if loader.group_probs_parquet_paths is not None:
        data.group_probs_parquet_paths = list(loader.group_probs_parquet_paths)
    return GroupSplitterDataset(data=data, targets=targets, loader=loader)
