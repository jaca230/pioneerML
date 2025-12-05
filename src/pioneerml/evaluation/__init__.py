"""
Evaluation metrics and diagnostic tools.

Includes:
- Metric registry with defaults for multilabel classification and regression
- Plot registry for standardized diagnostics
"""

from pioneerml.evaluation.metrics import (
    METRIC_REGISTRY,
    MetricCollection,
    default_metrics_for_task,
    multilabel_classification_metrics,
    regression_metrics,
    register_metric,
)
from pioneerml.evaluation.plots import (
    PLOT_REGISTRY,
    plot_embedding_space,
    plot_multilabel_confusion_matrix,
    plot_probability_distributions,
    plot_precision_recall_curves,
    plot_regression_diagnostics,
    plot_roc_curves,
    register_plot,
)
from pioneerml.evaluation.confidence import plot_confidence_analysis
from pioneerml.evaluation.utils import resolve_preds_targets

__all__ = [
    "METRIC_REGISTRY",
    "PLOT_REGISTRY",
    "MetricCollection",
    "default_metrics_for_task",
    "multilabel_classification_metrics",
    "regression_metrics",
    "register_metric",
    "plot_embedding_space",
    "plot_multilabel_confusion_matrix",
    "plot_precision_recall_curves",
    "plot_regression_diagnostics",
    "plot_roc_curves",
    "register_plot",
    "resolve_preds_targets",
    "plot_probability_distributions",
    "plot_confidence_analysis",
]
