"""
Built-in pipeline stages for common operations.

This package provides ready-to-use stages for typical ML workflows.
"""

from pioneerml.pipelines.stages.data import LoadDataStage, SaveDataStage
from pioneerml.pipelines.stages.model import TrainModelStage, InferenceStage, LightningTrainStage
from pioneerml.pipelines.stages.evaluation import EvaluateStage

__all__ = [
    "LoadDataStage",
    "SaveDataStage",
    "TrainModelStage",
    "InferenceStage",
    "LightningTrainStage",
    "EvaluateStage",
]
