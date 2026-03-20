from typing import Any

import torch.nn as nn
from zenml import step

from pioneerml.common.pipeline.steps import BaseFullTrainingStep


class EndpointRegressorTrainStep(BaseFullTrainingStep):
    step_key = "train"

    def default_config(self) -> dict:
        return {
            "loader": {
                "type": "endpoint_regression",
                "config": {},
            },
            "architecture": {
                "type": "endpoint_regressor",
                "config": {
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
            },
            "module": {
                "type": "graph_lightning",
                "config": {
                    "loss_fn": nn.MSELoss(),
                    "lr": 1e-3,
                    "weight_decay": 1e-4,
                    "scheduler_step_size": 2,
                    "scheduler_gamma": 0.5,
                },
            },
        }


@step(name="train_endpoint_regressor", enable_cache=False)
def train_endpoint_regressor_step(
    dataset,
    pipeline_config: dict | None = None,
    hpo_payload=None,
) -> Any:
    return EndpointRegressorTrainStep(pipeline_config=pipeline_config).execute(
        payloads={"loader": dataset, "hpo": hpo_payload}
    )
