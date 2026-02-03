"""
Example pipelines for PIONEER ML tutorials.

These pipelines demonstrate various ZenML patterns and can be used
in tutorial notebooks to show different concepts.
"""

from pioneerml.pipelines.tutorial_examples.dummy_particle_grouping_pipeline import (
    dummy_particle_grouping_pipeline,
)
from pioneerml.pipelines.tutorial_examples.dummy_particle_grouping_optuna_pipeline import (
    dummy_particle_grouping_optuna_pipeline,
)
from pioneerml.pipelines.tutorial_examples.quickstart_pipeline import quickstart_pipeline
from pioneerml.pipelines.tutorial_examples.tabular_datamodule_pipeline import (
    tabular_datamodule_pipeline,
    TabularConfig,
)

__all__ = [
    "dummy_particle_grouping_pipeline",
    "dummy_particle_grouping_optuna_pipeline",
    "quickstart_pipeline",
    "tabular_datamodule_pipeline",
    "TabularConfig",
]
