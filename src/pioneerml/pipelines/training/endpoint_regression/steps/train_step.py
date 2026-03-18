from typing import Any

from zenml import step

from pioneerml.common.pipeline.steps import BaseFullTrainingStep

from ..objective import EndpointRegressorObjectiveAdapter


class EndpointRegressorTrainStep(BaseFullTrainingStep):
    step_key = "train"

    def __init__(self, *, pipeline_config: dict | None = None) -> None:
        super().__init__(pipeline_config=pipeline_config)
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


@step(name="train_endpoint_regressor", enable_cache=False)
def train_endpoint_regressor_step(
    dataset,
    pipeline_config: dict | None = None,
    hpo_payload=None,
) -> Any:
    return EndpointRegressorTrainStep(pipeline_config=pipeline_config).execute(
        payloads={"loader": dataset, "hpo": hpo_payload}
    )
