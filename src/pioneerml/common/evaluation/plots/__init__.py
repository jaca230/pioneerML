"""
Plot utilities organized as one class per plot type.

Backwards-compatible function wrappers are provided to ease migration.
"""

from __future__ import annotations

from typing import Any, Callable, Dict

from .base import BasePlot
from .confusion import ConfusionMatrixPlot
from .loss import LossCurvesPlot
from .precision_recall import PrecisionRecallPlot
from .roc import RocCurvesPlot
from .regression import (
    RegressionDiagnosticsPlot,
    EuclideanErrorHistogramPlot,
    ErrorEmbeddingSpacePlot,
)
from .embedding import EmbeddingSpacePlot
from .probability import ProbabilityDistributionsPlot
from .confidence import ConfidenceAnalysisPlot

# Friendly aliases for discoverability
PLOT_CLASSES: Dict[str, type[BasePlot]] = {
    ConfusionMatrixPlot.name: ConfusionMatrixPlot,
    RocCurvesPlot.name: RocCurvesPlot,
    PrecisionRecallPlot.name: PrecisionRecallPlot,
    LossCurvesPlot.name: LossCurvesPlot,
    RegressionDiagnosticsPlot.name: RegressionDiagnosticsPlot,
    EuclideanErrorHistogramPlot.name: EuclideanErrorHistogramPlot,
    ErrorEmbeddingSpacePlot.name: ErrorEmbeddingSpacePlot,
    EmbeddingSpacePlot.name: EmbeddingSpacePlot,
    ProbabilityDistributionsPlot.name: ProbabilityDistributionsPlot,
    ConfidenceAnalysisPlot.name: ConfidenceAnalysisPlot,
}


def _wrap(cls: type[BasePlot]) -> Callable[..., Any]:
    def _fn(*args, **kwargs):
        return cls().render(*args, **kwargs)

    return _fn


# Back-compat registry and function aliases
PLOT_REGISTRY: Dict[str, Callable[..., Any]] = {name: _wrap(cls) for name, cls in PLOT_CLASSES.items()}

plot_multilabel_confusion_matrix = _wrap(ConfusionMatrixPlot)
plot_roc_curves = _wrap(RocCurvesPlot)
plot_precision_recall_curves = _wrap(PrecisionRecallPlot)
plot_loss_curves = _wrap(LossCurvesPlot)
plot_regression_diagnostics = _wrap(RegressionDiagnosticsPlot)
plot_euclidean_error_histogram = _wrap(EuclideanErrorHistogramPlot)
plot_error_embedding_space = _wrap(ErrorEmbeddingSpacePlot)
plot_embedding_space = _wrap(EmbeddingSpacePlot)
plot_probability_distributions = _wrap(ProbabilityDistributionsPlot)
plot_confidence_analysis = _wrap(ConfidenceAnalysisPlot)

__all__ = [
    "BasePlot",
    "PLOT_CLASSES",
    "PLOT_REGISTRY",
    "plot_multilabel_confusion_matrix",
    "plot_roc_curves",
    "plot_precision_recall_curves",
    "plot_loss_curves",
    "plot_regression_diagnostics",
    "plot_euclidean_error_histogram",
    "plot_error_embedding_space",
    "plot_embedding_space",
    "plot_probability_distributions",
    "plot_confidence_analysis",
    "ConfusionMatrixPlot",
    "RocCurvesPlot",
    "PrecisionRecallPlot",
    "LossCurvesPlot",
    "RegressionDiagnosticsPlot",
    "EuclideanErrorHistogramPlot",
    "ErrorEmbeddingSpacePlot",
    "EmbeddingSpacePlot",
    "ProbabilityDistributionsPlot",
    "ConfidenceAnalysisPlot",
]
