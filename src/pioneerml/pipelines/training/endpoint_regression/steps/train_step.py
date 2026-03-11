from typing import Any

from zenml import step

from pioneerml.common.data_loader import BatchBundle
from pioneerml.common.pipeline.steps import BaseFullTrainingStep

from ..objective import EndpointRegressorObjectiveAdapter


class EndpointRegressorTrainStep(BaseFullTrainingStep):
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
        self.objective_adapter = EndpointRegressorObjectiveAdapter()

    def default_config(self) -> dict:
        return {
            "lr": 1e-3,
            "weight_decay": 1e-4,
            "scheduler_step_size": 2,
            "scheduler_gamma": 0.5,
            "model": {
                "node_dim": 7,
                "graph_dim": 3,
                "splitter_prob_dimension": 0,
                "edge_dim": 4,
                "hidden": 192,
                "heads": 4,
                "layers": 3,
                "dropout": 0.1,
                "output_dim": 18,
            },
        }


@step(name="train_endpoint_regressor")
def train_endpoint_regressor_step(
    dataset: BatchBundle,
    pipeline_config: dict | None = None,
    hpo_params: dict | None = None,
) -> Any:
    return EndpointRegressorTrainStep(
        dataset=dataset,
        pipeline_config=pipeline_config,
        hpo_params=hpo_params,
    ).execute(payloads={"loader": dataset, "hpo_params": dict(hpo_params or {})})
