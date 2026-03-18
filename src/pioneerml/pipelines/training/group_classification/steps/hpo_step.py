from typing import Any

from zenml import step

from pioneerml.common.pipeline.steps import BaseHPOStep

from ..objective import GroupClassifierObjectiveAdapter


class GroupClassifierHPOStep(BaseHPOStep):
    step_key = "hpo"

    def default_config(self) -> dict:
        return {
            "n_trials": 5,
            "max_epochs": 3,
            "grad_clip": 2.0,
            "trainer_kwargs": {"enable_progress_bar": True},
            "batch_size": {"min_exp": 5, "max_exp": 7},
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
            "compile": {"enabled": False, "mode": "default"},
            "direction": "minimize",
            "seed": None,
            "study_name": "group_classifier_hpo",
            "storage": None,
            "fallback_dir": None,
            "allow_schema_fallback": True,
            "model": {
                "hidden": {"low": 64, "high": 256, "log": False},
                "heads": {"low": 2, "high": 8, "log": False},
                "num_blocks": {"low": 1, "high": 4, "log": False},
                "dropout": {"low": 0.0, "high": 0.3, "log": False},
            },
            "train": {
                "lr": {"low": 1e-4, "high": 1e-2, "log": True},
                "weight_decay": {"low": 1e-6, "high": 1e-3, "log": True},
                "scheduler_step_size": 2,
                "scheduler_gamma": 0.5,
                "threshold": 0.5,
            },
        }

    def build_objective_adapter(self):
        return GroupClassifierObjectiveAdapter()


@step(name="tune_group_classifier", enable_cache=False)
def tune_group_classifier_step(
    dataset,
    pipeline_config: dict | None = None,
) -> Any:
    return GroupClassifierHPOStep(pipeline_config=pipeline_config).execute(payloads={"loader": dataset})
