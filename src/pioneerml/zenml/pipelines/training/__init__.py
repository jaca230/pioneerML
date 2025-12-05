"""
Training pipelines for real data.

These pipelines load preprocessed time-group data from .npy files and train
models using ZenML with Optuna hyperparameter tuning.
"""

from pioneerml.zenml.pipelines.training.group_classification_pipeline import (
    group_classification_optuna_pipeline,
)
from pioneerml.zenml.pipelines.training.pion_stop_pipeline import (
    pion_stop_optuna_pipeline,
)
from pioneerml.zenml.pipelines.training.group_splitter_pipeline import (
    group_splitter_optuna_pipeline,
)
from pioneerml.zenml.pipelines.training.positron_angle_pipeline import (
    positron_angle_optuna_pipeline,
)

__all__ = [
    "group_classification_optuna_pipeline",
    "pion_stop_optuna_pipeline",
    "group_splitter_optuna_pipeline",
    "positron_angle_optuna_pipeline",
]

