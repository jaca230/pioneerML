from typing import Any

from zenml import step

from pioneerml.common.pipeline.steps import BaseEvaluationStep


class GroupClassifierEvaluateStep(BaseEvaluationStep):
    step_key = "evaluate"

    def default_config(self) -> dict:
        return {
            "evaluator_name": "simple_classification",
            "threshold": 0.5,
            "metrics": ["binary_classification_from_tensors"],
            "plots": ["loss_curves"],
            "loader_config": {
                "base": {
                    "batch_size": 1,
                    "chunk_row_groups": 4,
                    "chunk_workers": 0,
                },
                "test": {"mode": "train", "split": "test", "shuffle_batches": False, "log_diagnostics": False},
            },
        }


@step(name="evaluate_group_classifier", enable_cache=False)
def evaluate_group_classifier_step(
    train_payload,
    dataset,
    pipeline_config: dict | None = None,
) -> Any:
    return GroupClassifierEvaluateStep(pipeline_config=pipeline_config).execute(
        payloads={"loader": dataset, "train": train_payload}
    )
