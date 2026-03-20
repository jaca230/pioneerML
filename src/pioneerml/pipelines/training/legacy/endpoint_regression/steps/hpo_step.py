from typing import Any

import torch.nn as nn
from zenml import step

from pioneerml.common.pipeline.steps import BaseHPOStep


class EndpointRegressorHPOStep(BaseHPOStep):
    step_key = "hpo"

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
            "trainer": {
                "type": "lightning_module",
                "config": {
                    "trainer_kwargs": {
                        "enable_progress_bar": True,
                        "max_epochs": 3,
                        "gradient_clip_val": 2.0,
                    },
                    "early_stopping": {
                        "enabled": False,
                        "type": "relative",
                        "config": {
                            "monitor": "val_loss",
                            "mode": "min",
                            "patience": 3,
                            "min_delta": 0.0,
                            "strict": True,
                            "check_finite": True,
                            "verbose": False,
                        },
                    },
                },
            },
            "hpo": {
                "type": "config",
                "config": {
                    "n_trials": 5,
                    "direction": "minimize",
                    "seed": None,
                    "study_name": "endpoint_regressor_hpo",
                    "storage": None,
                    "fallback_dir": None,
                    "allow_schema_fallback": True,
                    "objective": {"type": "val_epoch", "config": {}},
                    "search_space": {
                        "type": "config",
                        "config": {
                            "search_space": {
                                "batch_size": {"type": "exponent_int", "base": 2, "min_exp": 5, "max_exp": 7},
                                "lr": {"type": "float", "low": 1e-4, "high": 1e-2, "log": True},
                                "weight_decay": {"type": "float", "low": 1e-6, "high": 1e-3, "log": True},
                                "hidden": {"type": "int", "low": 96, "high": 320},
                                "heads": {"type": "int", "low": 2, "high": 8},
                                "layers": {"type": "int", "low": 1, "high": 4},
                                "dropout": {"type": "float", "low": 0.0, "high": 0.3},
                            }
                        },
                    },
                },
            },
        }


@step(name="tune_endpoint_regressor", enable_cache=False)
def tune_endpoint_regressor_step(
    dataset,
    pipeline_config: dict | None = None,
) -> Any:
    return EndpointRegressorHPOStep(pipeline_config=pipeline_config).execute(payloads={"loader": dataset})
