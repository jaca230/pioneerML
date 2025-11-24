"""
Built-in pipeline stages for common operations.

This package provides ready-to-use stages for typical ML workflows.
"""

from pioneerml.pipelines.stages.providers.load_data import LoadDataStage
from pioneerml.pipelines.stages.providers.save_data import SaveDataStage
from pioneerml.pipelines.stages.trainers.train_model import TrainModelStage
from pioneerml.pipelines.stages.trainers.lightning_train import LightningTrainStage
from pioneerml.pipelines.stages.collectors.collect_preds import CollectPredsStage
from pioneerml.pipelines.stages.evaluators.evaluation import EvaluateStage
from pioneerml.pipelines.stages.runners.inference import InferenceStage
from pioneerml.pipelines.stages.roles import (
    ProviderStage,
    TrainerStage,
    CollectorStage,
    EvaluatorStage,
    RunnerStage,
)

__all__ = [
    "ProviderStage",
    "TrainerStage",
    "CollectorStage",
    "EvaluatorStage",
    "RunnerStage",
    "LoadDataStage",
    "SaveDataStage",
    "TrainModelStage",
    "InferenceStage",
    "LightningTrainStage",
    "CollectPredsStage",
    "EvaluateStage",
]

# Simple registries for hotswapping by role
PROVIDERS = {
    "load_data": LoadDataStage,
    "save_data": SaveDataStage,
}

TRAINERS = {
    "lightning": LightningTrainStage,
    "torch": TrainModelStage,
}

COLLECTORS = {
    "preds": CollectPredsStage,
}

EVALUATORS = {
    "default": EvaluateStage,
}

RUNNERS = {
    "default": InferenceStage,
}
