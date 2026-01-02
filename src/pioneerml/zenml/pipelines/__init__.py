"""
ZenML pipelines for PIONEER ML.

All example pipelines used in tutorials live under the ``tutorial_examples``
module to keep educational flows organized in one place.
"""

from pioneerml.zenml.pipelines.tutorial_examples.dummy_particle_grouping_pipeline import (
    dummy_particle_grouping_pipeline,
)
from pioneerml.zenml.pipelines.tutorial_examples.dummy_particle_grouping_optuna_pipeline import (
    dummy_particle_grouping_optuna_pipeline,
)
from pioneerml.zenml.pipelines.tutorial_examples.quickstart_pipeline import quickstart_pipeline
from pioneerml.zenml.pipelines.training import (
    group_classification_optuna_pipeline,
    pion_stop_optuna_pipeline,
    group_splitter_optuna_pipeline,
)
from pioneerml.zenml.pipelines.inference import (
    upstream_inference_pipeline,
    downstream_inference_pipeline,
)

__all__ = [
    "dummy_particle_grouping_pipeline",
    "dummy_particle_grouping_optuna_pipeline",
    "quickstart_pipeline",
    "group_classification_optuna_pipeline",
    "pion_stop_optuna_pipeline",
    "group_splitter_optuna_pipeline",
    "upstream_inference_pipeline",
    "downstream_inference_pipeline",
]
