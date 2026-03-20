from typing import Any

import torch.nn as nn
from zenml import step

from pioneerml.common.pipeline.steps import BaseFullTrainingStep


class GroupSplitterTrainStep(BaseFullTrainingStep):
    step_key = "train"

    def default_config(self) -> dict:
        return {
            "loader": {
                "type": "group_splitter",
                "config": {},
            },
            "architecture": {
                "type": "group_splitter",
                "config": {
                    "node_dim": 4,
                    "edge_dim": 4,
                    "graph_dim": 3,
                    "hidden": 200,
                    "heads": 4,
                    "layers": 3,
                    "dropout": 0.1,
                },
            },
            "module": {
                "type": "graph_lightning",
                "config": {
                    "loss_fn": nn.BCEWithLogitsLoss(),
                    "lr": 1e-3,
                    "weight_decay": 1e-4,
                    "scheduler_step_size": 2,
                    "scheduler_gamma": 0.5,
                },
            },
        }


@step(name="train_group_splitter", enable_cache=False)
def train_group_splitter_step(
    dataset,
    pipeline_config: dict | None = None,
    hpo_payload=None,
) -> Any:
    return GroupSplitterTrainStep(pipeline_config=pipeline_config).execute(
        payloads={"loader": dataset, "hpo": hpo_payload}
    )
