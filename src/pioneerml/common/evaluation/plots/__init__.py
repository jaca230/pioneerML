from typing import Any, Callable

from .base_plot import BasePlot
from .classification import (
    ConfidenceAnalysisPlot,
    ConfusionMatrixPlot,
    EmbeddingSpacePlot,
    PrecisionRecallPlot,
    ProbabilityDistributionsPlot,
    RocCurvesPlot,
)
from .factory import PlotFactory
from .loss import LossCurvesPlot
from .regression import ErrorEmbeddingSpacePlot, EuclideanErrorHistogramPlot, RegressionDiagnosticsPlot
from .registry import PLOT_REGISTRY, REGISTRY as PLOT_PLUGIN_REGISTRY

PLOT_CLASSES = PLOT_REGISTRY


def _wrap(cls: type[BasePlot]) -> Callable[..., Any]:
    def _fn(*args, **kwargs):
        return cls().render(*args, **kwargs)

    return _fn


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
    "PlotFactory",
    "PLOT_PLUGIN_REGISTRY",
    "PLOT_REGISTRY",
    "PLOT_CLASSES",
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
