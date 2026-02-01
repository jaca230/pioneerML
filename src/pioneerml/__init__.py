"""
PIONEER ML: Machine Learning Pipeline Framework for PIONEER Experiment AI Reconstruction.

This package provides Graph Neural Network models and training utilities for
reconstructing particle physics events from the PIONEER Active Target detector.
"""

__version__ = "0.1.0"
__author__ = "Jack"

from pioneerml import models, training, evaluation, utils, zenml

__all__ = [
    "models",
    "training",
    "evaluation",
    "utils",
    "zenml",
    "__version__",
]
