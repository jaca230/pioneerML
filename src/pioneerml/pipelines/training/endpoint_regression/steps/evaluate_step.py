from typing import Any

from zenml import step

from pioneerml.common.pipeline.steps import BaseEvaluationStep


class EndpointRegressorEvaluateStep(BaseEvaluationStep):
    step_key = "evaluate"

    def default_config(self) -> dict:
        return {
            "evaluator_name": "simple_regression",
            "plots": ["loss_curves"],
            "loader_config": {
                "base": {
                    "batch_size": 64,
                    "chunk_row_groups": 4,
                    "chunk_workers": 0,
                },
                "test": {"mode": "train", "split": "test", "shuffle_batches": False, "log_diagnostics": False},
            },
        }


@step(name="evaluate_endpoint_regressor", enable_cache=False)
def evaluate_endpoint_regressor_step(
    train_payload,
    dataset,
    pipeline_config: dict | None = None,
) -> Any:
    return EndpointRegressorEvaluateStep(pipeline_config=pipeline_config).execute(
        payloads={"loader": dataset, "train": train_payload}
    )
