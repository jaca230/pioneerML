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
    build_registry,
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
    "resolve_stage",
    "list_stage_names",
]

# Build registries from subclasses (name attribute on each subclass)
PROVIDERS = build_registry(ProviderStage)
TRAINERS = build_registry(TrainerStage)
COLLECTORS = build_registry(CollectorStage)
EVALUATORS = build_registry(EvaluatorStage)
RUNNERS = build_registry(RunnerStage)


def resolve_stage(role: str, name: str, config) -> object:
    """Instantiate a stage by role/name using the registries."""
    registry_map = {
        "provider": PROVIDERS,
        "trainer": TRAINERS,
        "collector": COLLECTORS,
        "evaluator": EVALUATORS,
        "runner": RUNNERS,
    }
    registry = registry_map.get(role)
    if registry is None:
        raise KeyError(f"Unknown role '{role}'.")
    if name not in registry:
        raise KeyError(f"Stage '{name}' not found for role '{role}'. Available: {list(registry)}")
    return registry[name](config=config)


def list_stage_names(role: str) -> list[str]:
    """List available stage names for a role."""
    registry_map = {
        "provider": PROVIDERS,
        "trainer": TRAINERS,
        "collector": COLLECTORS,
        "evaluator": EVALUATORS,
        "runner": RUNNERS,
    }
    registry = registry_map.get(role)
    if registry is None:
        return []
    return sorted(registry.keys())
