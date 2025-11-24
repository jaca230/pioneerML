"""
Role-specific Stage base classes for hotswappable components.

These subclasses of Stage encode the intended role (provider/trainer/collector/evaluator/runner)
so registries and type hints can be clearer without changing runtime behavior.
"""

from __future__ import annotations

from pioneerml.pipelines.stage import Stage


class ProviderStage(Stage):
    """Base class for stages that provide data to the pipeline (loaders, savers)."""

    role = "provider"


class TrainerStage(Stage):
    """Base class for stages that train models."""

    role = "trainer"


class CollectorStage(Stage):
    """Base class for stages that collect predictions/targets or features."""

    role = "collector"


class EvaluatorStage(Stage):
    """Base class for stages that compute metrics/plots/reports."""

    role = "evaluator"


class RunnerStage(Stage):
    """Base class for stages that run inference or deployment-time workloads."""

    role = "runner"
