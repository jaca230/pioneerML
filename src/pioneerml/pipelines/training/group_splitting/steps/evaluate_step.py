from typing import Any

from zenml import step

from pioneerml.common.pipeline.steps import BaseEvaluationStep


class GroupSplitterEvaluateStep(BaseEvaluationStep):
    step_key = "evaluate"

    def default_config(self) -> dict:
        return {
            "evaluator": {
                "type": "simple_classification",
                "config": {
                    "threshold": 0.5,
                },
            },
            "metrics": ["binary_classification_from_tensors"],
            "plots": ["loss_curves"],
            "loader": {
                "type": "group_splitter",
                "config": {
                    "base": {
                        "batch_size": 1,
                        "chunk_row_groups": 4,
                        "chunk_workers": 0,
                    },
                    "test": {"mode": "train", "split": "test", "shuffle_batches": False, "log_diagnostics": False},
                },
            },
        }


@step(name="evaluate_group_splitter", enable_cache=False)
def evaluate_group_splitter_step(
    train_payload,
    dataset,
    pipeline_config: dict | None = None,
) -> Any:
    return GroupSplitterEvaluateStep(pipeline_config=pipeline_config).execute(
        payloads={"loader": dataset, "train": train_payload}
    )
