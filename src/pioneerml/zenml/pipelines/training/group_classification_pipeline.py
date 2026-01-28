"""
Backwards-compatible entrypoint that re-exports the modular group classification pipeline.

New code lives in src/pioneerml/zenml/pipelines/training/group_classification/.
"""

from .group_classification import (
    group_classification_pipeline,
    group_classification_optuna_pipeline,
    build_group_classification_datamodule_step as build_group_classification_datamodule,
    run_group_classification_hparam_search,
    train_best_group_classifier,
    collect_group_classification_predictions,
    GroupClassificationProcessor,
)

__all__ = [
    "group_classification_pipeline",
    "group_classification_optuna_pipeline",
    "build_group_classification_datamodule",
    "run_group_classification_hparam_search",
    "train_best_group_classifier",
    "collect_group_classification_predictions",
    "GroupClassificationProcessor",
]
