
from typing import Any

from zenml import step

from pioneerml.common.data_loader import BatchBundle
from pioneerml.common.pipeline.steps import BaseFullTrainingStep

from ..objective import GroupClassifierObjectiveAdapter


class GroupClassifierTrainStep(BaseFullTrainingStep):
    step_key = "train"

    def __init__(
        self,
        *,
        dataset: BatchBundle,
        pipeline_config: dict | None = None,
        hpo_params: dict | None = None,
    ) -> None:
        super().__init__(pipeline_config=pipeline_config)
        self.dataset = dataset
        self.hpo_params = dict(hpo_params or {})
        self.objective_adapter = GroupClassifierObjectiveAdapter()

    def default_config(self) -> dict:
        return {
            "lr": 1e-3,
            "weight_decay": 1e-4,
            "scheduler_step_size": 2,
            "scheduler_gamma": 0.5,
            "threshold": 0.5,
            "model": {
                "node_dim": 4,
                "edge_dim": 4,
                "graph_dim": 0,
                "hidden": 200,
                "heads": 4,
                "num_blocks": 2,
                "dropout": 0.1,
            },
        }


@step(name="train_group_classifier")
def train_group_classifier_step(
    dataset: BatchBundle,
    pipeline_config: dict | None = None,
    hpo_params: dict | None = None,
) -> Any:
    return GroupClassifierTrainStep(
        dataset=dataset,
        pipeline_config=pipeline_config,
        hpo_params=hpo_params,
    ).execute(payloads={"loader": dataset, "hpo_params": dict(hpo_params or {})})
