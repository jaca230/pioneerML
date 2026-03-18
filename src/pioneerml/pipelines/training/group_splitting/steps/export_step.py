from typing import Any

from zenml import step

from pioneerml.common.pipeline.steps import BaseExportStep


class GroupSplitterExportStep(BaseExportStep):
    step_key = "export"

    def default_config(self) -> dict:
        return {
            "export_type": "script",
            "export_dir": "trained_models/groupsplitter",
            "filename_prefix": "groupsplitter",
            "prefer_cuda": True,
            "loader_config": {
                "base": {"batch_size": 1, "chunk_row_groups": 1, "chunk_workers": 0},
                "export": {"mode": "train", "shuffle_batches": False, "log_diagnostics": False},
            },
        }


@step(name="export_group_splitter", enable_cache=False)
def export_group_splitter_step(
    train_payload,
    dataset,
    pipeline_config: dict | None = None,
    hpo_payload=None,
    metrics=None,
) -> Any:
    return GroupSplitterExportStep(pipeline_config=pipeline_config).execute(
        payloads={
            "train": train_payload,
            "loader": dataset,
            "hpo": hpo_payload,
            "metrics": metrics,
        }
    )
