"""Evaluation package: evaluators, metric registry, and plot registry."""

from pioneerml.common.evaluation.evaluators import (
    BaseClassificationEvaluator,
    BaseEvaluator,
    BaseRegressionEvaluator,
    EvaluatorFactory,
    SimpleClassificationEvaluator,
    SimpleRegressionEvaluator,
    list_registered_evaluators,
    register_evaluator,
    resolve_evaluator,
)
from pioneerml.common.evaluation.metrics import (
    METRIC_REGISTRY,
    BaseMetric,
    compute_metrics,
    create_metric,
    register_metric,
)
from pioneerml.common.evaluation.plots import (
    PLOT_CLASSES,
    PLOT_REGISTRY,
    BasePlot,
    create_plot,
    register_plot,
    render_plots,
)

__all__ = [
    "BaseEvaluator",
    "EvaluatorFactory",
    "register_evaluator",
    "resolve_evaluator",
    "list_registered_evaluators",
    "BaseClassificationEvaluator",
    "BaseRegressionEvaluator",
    "SimpleClassificationEvaluator",
    "SimpleRegressionEvaluator",
    "BaseMetric",
    "METRIC_REGISTRY",
    "register_metric",
    "create_metric",
    "compute_metrics",
    "BasePlot",
    "PLOT_REGISTRY",
    "PLOT_CLASSES",
    "register_plot",
    "create_plot",
    "render_plots",
]
