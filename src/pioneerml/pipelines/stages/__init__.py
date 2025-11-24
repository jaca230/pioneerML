"""
Built-in pipeline stages for common operations.

This package provides ready-to-use stages for typical ML workflows.
"""

from pioneerml.pipelines.stages.data.load_data import LoadDataStage
from pioneerml.pipelines.stages.data.save_data import SaveDataStage
from pioneerml.pipelines.stages.training.train_model import TrainModelStage
from pioneerml.pipelines.stages.training.lightning_train import LightningTrainStage
from pioneerml.pipelines.stages.inference.inference import InferenceStage
from pioneerml.pipelines.stages.evaluation.collect_preds import CollectPredsStage
from pioneerml.pipelines.stages.evaluation.evaluation import EvaluateStage

__all__ = [
    "LoadDataStage",
    "SaveDataStage",
    "TrainModelStage",
    "InferenceStage",
    "LightningTrainStage",
    "CollectPredsStage",
    "EvaluateStage",
]
