"""Evaluation metrics and diagnostic tools."""

from pioneerml.common.evaluation.metrics import (
    METRIC_REGISTRY,
    MetricCollection,
    default_metrics_for_task,
    multilabel_classification_metrics,
    regression_metrics,
    register_metric,
)
from pioneerml.common.evaluation.plots import (
    PLOT_CLASSES,
    PLOT_REGISTRY,
    plot_embedding_space,
    plot_multilabel_confusion_matrix,
    plot_probability_distributions,
    plot_precision_recall_curves,
    plot_regression_diagnostics,
    plot_roc_curves,
    plot_confidence_analysis,
)
from pioneerml.common.evaluation.utils import resolve_preds_targets

__all__ = [
    "METRIC_REGISTRY",
    "PLOT_CLASSES",
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
    "resolve_preds_targets",
    "plot_probability_distributions",
    "plot_confidence_analysis",
]
