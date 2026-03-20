"""Evaluation package: evaluators, metric registry, and plot registry."""

from pioneerml.common.evaluation.evaluators import (
    BaseClassificationEvaluator,
    BaseEvaluator,
    BaseRegressionEvaluator,
    EVALUATOR_REGISTRY,
    EvaluatorFactory,
    SimpleClassificationEvaluator,
    SimpleRegressionEvaluator,
)
from pioneerml.common.evaluation.metrics import (
    METRIC_PLUGIN_REGISTRY,
    METRIC_REGISTRY,
    BaseMetric,
    MetricFactory,
)
from pioneerml.common.evaluation.plots import (
    PLOT_CLASSES,
    PLOT_PLUGIN_REGISTRY,
    PLOT_REGISTRY,
    BasePlot,
    PlotFactory,
)

__all__ = [
    "BaseEvaluator",
    "EvaluatorFactory",
    "EVALUATOR_REGISTRY",
    "BaseClassificationEvaluator",
    "BaseRegressionEvaluator",
    "SimpleClassificationEvaluator",
    "SimpleRegressionEvaluator",
    "BaseMetric",
    "MetricFactory",
    "METRIC_PLUGIN_REGISTRY",
    "METRIC_REGISTRY",
    "BasePlot",
    "PlotFactory",
    "PLOT_PLUGIN_REGISTRY",
    "PLOT_REGISTRY",
    "PLOT_CLASSES",
]
