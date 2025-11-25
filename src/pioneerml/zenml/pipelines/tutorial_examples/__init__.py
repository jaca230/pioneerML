"""
Example pipelines for PIONEER ML tutorials.

These pipelines demonstrate various ZenML patterns and can be used
in tutorial notebooks to show different concepts.
"""

from pioneerml.zenml.pipelines.tutorial_examples.basic_training import basic_training_pipeline
from pioneerml.zenml.pipelines.tutorial_examples.custom_model import custom_model_pipeline
from pioneerml.zenml.pipelines.tutorial_examples.evaluation_examples import evaluation_examples_pipeline
from pioneerml.zenml.pipelines.tutorial_examples.hyperparameter_tuning import hyperparameter_tuning_pipeline
from pioneerml.zenml.pipelines.tutorial_examples.training_pipeline import zenml_training_pipeline

__all__ = [
    "basic_training_pipeline",
    "custom_model_pipeline",
    "evaluation_examples_pipeline",
    "hyperparameter_tuning_pipeline",
    "zenml_training_pipeline",
]
