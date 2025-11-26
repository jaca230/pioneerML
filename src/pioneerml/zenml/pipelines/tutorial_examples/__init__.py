"""
Example pipelines for PIONEER ML tutorials.

These pipelines demonstrate various ZenML patterns and can be used
in tutorial notebooks to show different concepts.
"""

from pioneerml.zenml.pipelines.tutorial_examples.quickstart_pipeline import quickstart_pipeline
from pioneerml.zenml.pipelines.tutorial_examples.tabular_datamodule_pipeline import (
    tabular_datamodule_pipeline,
    TabularConfig,
)

__all__ = [
    "quickstart_pipeline",
    "tabular_datamodule_pipeline",
    "TabularConfig",
]
