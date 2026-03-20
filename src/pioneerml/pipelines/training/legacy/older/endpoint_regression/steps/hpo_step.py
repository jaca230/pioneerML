from zenml import step

from pioneerml.common.data_loader import BatchBundle
from pioneerml.common.pipeline.steps import BaseHPOStep

from ..objective import EndpointRegressorObjectiveAdapter


class EndpointRegressorHPOStep(BaseHPOStep):
    step_key = "hpo"

    def default_config(self) -> dict:
        return {
            "n_trials": 5,
            "max_epochs": 3,
            "grad_clip": 2.0,
            "trainer_kwargs": {"enable_progress_bar": True},
            "batch_size": {"min_exp": 5, "max_exp": 7},
            "chunk_row_groups": 4,
            "chunk_workers": None,
            "max_train_batches": None,
            "max_val_batches": None,
            "early_stopping": {
                "enabled": False,
                "monitor": "val_loss",
                "mode": "min",
                "patience": 3,
                "min_delta": 0.0,
                "min_delta_mode": "absolute",
            },
            "compile": {"enabled": False, "mode": "default"},
            "direction": "minimize",
            "seed": None,
            "study_name": "endpoint_regressor_hpo",
            "storage": None,
            "fallback_dir": None,
            "allow_schema_fallback": True,
            "model": {
                "hidden": {"low": 96, "high": 320, "log": False},
                "heads": {"low": 2, "high": 8, "log": False},
                "layers": {"low": 1, "high": 4, "log": False},
                "dropout": {"low": 0.0, "high": 0.3, "log": False},
            },
            "train": {
                "lr": {"low": 1e-4, "high": 1e-2, "log": True},
                "weight_decay": {"low": 1e-6, "high": 1e-3, "log": True},
                "scheduler_step_size": 2,
                "scheduler_gamma": 0.5,
            },
        }

    def build_objective_adapter(self):
        return EndpointRegressorObjectiveAdapter()


@step(name="tune_endpoint_regressor")
def tune_endpoint_regressor_step(
    dataset: BatchBundle,
    pipeline_config: dict | None = None,
) -> dict:
    return EndpointRegressorHPOStep(pipeline_config=pipeline_config).execute(
        payloads={"loader": dataset}
    )
