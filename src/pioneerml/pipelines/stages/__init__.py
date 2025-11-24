"""
Built-in pipeline stages for common operations.

This package provides ready-to-use stages for typical ML workflows.
"""

from pioneerml.pipelines.stages.load_data import LoadDataStage
from pioneerml.pipelines.stages.save_data import SaveDataStage
from pioneerml.pipelines.stages.train_model import TrainModelStage
from pioneerml.pipelines.stages.inference import InferenceStage
from pioneerml.pipelines.stages.lightning_train import LightningTrainStage
from pioneerml.pipelines.stages.collect_preds import CollectPredsStage
from pioneerml.pipelines.stages.evaluation import EvaluateStage

__all__ = [
    "LoadDataStage",
    "SaveDataStage",
    "TrainModelStage",
    "InferenceStage",
    "LightningTrainStage",
    "CollectPredsStage",
    "EvaluateStage",
]
