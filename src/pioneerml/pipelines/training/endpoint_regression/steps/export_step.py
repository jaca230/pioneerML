from typing import Any

from zenml import step

from pioneerml.common.pipeline.steps import BaseExportStep


class EndpointRegressorExportStep(BaseExportStep):
    step_key = "export"

    def default_config(self) -> dict:
        return {
            "export_type": "script",
            "export_dir": "trained_models/endpoint_regressor",
            "filename_prefix": "endpoint_regressor",
            "prefer_cuda": True,
            "loader_config": {
                "base": {"batch_size": 1, "chunk_row_groups": 1, "chunk_workers": 0},
                "export": {"mode": "train", "shuffle_batches": False, "log_diagnostics": False},
            },
        }


@step(name="export_endpoint_regressor", enable_cache=False)
def export_endpoint_regressor_step(
    train_payload,
    dataset,
    pipeline_config: dict | None = None,
    hpo_payload=None,
    metrics=None,
) -> Any:
    return EndpointRegressorExportStep(pipeline_config=pipeline_config).execute(
        payloads={
            "train": train_payload,
            "loader": dataset,
            "hpo": hpo_payload,
            "metrics": metrics,
        }
    )
