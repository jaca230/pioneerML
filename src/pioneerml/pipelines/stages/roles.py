"""
Role-specific Stage base classes for hotswappable components.

These subclasses of Stage encode the intended role (provider/trainer/collector/evaluator/runner)
so registries and type hints can be clearer without changing runtime behavior.
"""

from __future__ import annotations

from pioneerml.pipelines.stage import Stage


def _default_name(cls: type[Stage]) -> str:
    """Generate a default registry name based on class name."""
    return cls.__name__.replace("Stage", "").lower()


class ProviderStage(Stage):
    """Base class for stages that provide data to the pipeline (loaders, savers)."""

    role = "provider"
    name = "provider"

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        if not hasattr(cls, "name"):
            cls.name = _default_name(cls)


class TrainerStage(Stage):
    """Base class for stages that train models."""

    role = "trainer"
    name = "trainer"

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        if not hasattr(cls, "name"):
            cls.name = _default_name(cls)


class CollectorStage(Stage):
    """Base class for stages that collect predictions/targets or features."""

    role = "collector"
    name = "collector"

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        if not hasattr(cls, "name"):
            cls.name = _default_name(cls)


class EvaluatorStage(Stage):
    """Base class for stages that compute metrics/plots/reports."""

    role = "evaluator"
    name = "evaluator"

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        if not hasattr(cls, "name"):
            cls.name = _default_name(cls)


class RunnerStage(Stage):
    """Base class for stages that run inference or deployment-time workloads."""

    role = "runner"
    name = "runner"

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        if not hasattr(cls, "name"):
            cls.name = _default_name(cls)


def iter_subclasses(base: type[Stage]):
    """Recursively yield subclasses of a base Stage class."""
    for subclass in base.__subclasses__():
        yield subclass
        yield from iter_subclasses(subclass)


def build_registry(base: type[Stage]) -> dict[str, type[Stage]]:
    """Build a name->class registry for subclasses of a base role."""
    registry: dict[str, type[Stage]] = {}
    for cls in iter_subclasses(base):
        name = getattr(cls, "name", _default_name(cls))
        registry[name] = cls
    return registry
