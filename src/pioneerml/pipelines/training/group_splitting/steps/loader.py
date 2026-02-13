from zenml import step

from pioneerml.pipelines.training.group_splitting.dataset import GroupSplitterDataset


@step
def load_group_splitter_dataset(
    parquet_paths: list[str],
    group_probs_parquet_paths: list[str] | None = None,
    pipeline_config: dict | None = None,
) -> GroupSplitterDataset:
    raise NotImplementedError(
        "load_group_splitter_dataset was moved to deprecated for C++-loader retirement. "
        "Implement the new pure-Python/PyG loader for this pipeline."
    )
