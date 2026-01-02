"""
PIONEER ML: Machine Learning Pipeline Framework for PIONEER Experiment AI Reconstruction.

This package provides Graph Neural Network models and training utilities for
reconstructing particle physics events from the PIONEER Active Target detector.
"""

__version__ = "0.1.0"
__author__ = "Jack"

from pioneerml import models, data, training, evaluation, utils, pipelines

__all__ = [
    "models",
    "data",
    "training",
    "evaluation",
    "utils",
    "pipelines",
    "__version__",
]
