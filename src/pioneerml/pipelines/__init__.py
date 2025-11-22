"""
Pipeline framework for composable ML workflows.

This module implements a flexible DAG-based pipeline system where:
- Stages can be arbitrary processing nodes (preprocessing, models, post-processing)
- Stages connect via edges defining data flow
- Shared state/memory allows stages to communicate
- Supports both training pipelines and inference pipelines

Philosophy:
- No prescriptive stage types (no assumption of "preprocessor" → "classifier" → etc.)
- Graph-based execution with dependency resolution
- Stages are self-contained with clear inputs/outputs
- Easy to swap, add, or remove stages
"""

from pioneerml.pipelines.stage import Stage, StageConfig, FunctionalStage
from pioneerml.pipelines.context import Context
from pioneerml.pipelines.pipeline import Pipeline

__all__ = [
    "Stage",
    "StageConfig",
    "FunctionalStage",
    "Context",
    "Pipeline",
]
