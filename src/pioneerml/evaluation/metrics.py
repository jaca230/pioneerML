"""
Standardized metrics for classification and regression tasks.

Provides:
- Metric registry for easy extension by collaborators
- Default metric sets for multilabel classification (bitmask) and regression
- Helper to combine multiple metrics into a single computation
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterable, Mapping, Sequence

import numpy as np
import torch
from sklearn.metrics import (
    average_precision_score,
    mean_absolute_error,
    mean_squared_error,
    precision_recall_fscore_support,
    r2_score,
    roc_auc_score,
)

MetricFn = Callable[[Any, Any], Dict[str, float]]


def _to_numpy(arr: Any) -> np.ndarray:
    """Convert torch/numpy/sequence inputs to a numpy array."""
    if arr is None:
        raise ValueError("Targets are required for metric computation.")
    if torch.is_tensor(arr):
        return arr.detach().cpu().numpy()
    if isinstance(arr, (list, tuple)):
        return np.asarray(arr)
    if isinstance(arr, np.ndarray):
        return arr
    raise TypeError(f"Unsupported input type for metrics: {type(arr)}")


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def _normalize_probs(scores: np.ndarray) -> np.ndarray:
    """Ensure scores are in [0, 1] by applying sigmoid if needed."""
    if scores.min() < 0.0 or scores.max() > 1.0:
        return _sigmoid(scores)
    return scores


METRIC_REGISTRY: dict[str, MetricFn] = {}


def register_metric(name: str) -> Callable[[MetricFn], MetricFn]:
    """Decorator to register a metric function by name."""

    def decorator(fn: MetricFn) -> MetricFn:
        METRIC_REGISTRY[name] = fn
        return fn

    return decorator


@register_metric("multilabel_classification")
def multilabel_classification_metrics(
    predictions: Any,
    targets: Any,
    *,
    threshold: float = 0.5,
    average: str = "macro",
    class_names: Sequence[str] | None = None,
) -> Dict[str, float]:
    """
    Compute multi-label classification metrics for bitmask-style outputs.

    Args:
        predictions: Logits or probabilities with shape [N, C].
        targets: Binary targets with shape [N, C].
        threshold: Decision threshold applied to probabilities.
        average: Averaging strategy for sklearn metrics.
        class_names: Optional names for classwise metrics.
    """
    y_true = _to_numpy(targets)
    y_score = _normalize_probs(_to_numpy(predictions))

    if y_true.ndim == 1:
        y_true = y_true.reshape(-1, 1)
    if y_score.ndim == 1:
        y_score = y_score.reshape(-1, 1)

    # If total elements match but shapes differ, align targets to predictions
    if y_true.size == y_score.size and y_true.shape != y_score.shape:
        y_true = y_true.reshape(y_score.shape)

    # Align class names
    num_classes = y_score.shape[1]
    labels = list(class_names) if class_names is not None else [str(i) for i in range(num_classes)]
    if len(labels) < num_classes:
        labels.extend(str(i) for i in range(len(labels), num_classes))

    if y_true.shape != y_score.shape:
        raise ValueError(
            f"Predictions and targets shapes differ and cannot be aligned: preds={y_score.shape}, targets={y_true.shape}"
        )

    y_pred = (y_score >= threshold).astype(int)
    metrics: Dict[str, float] = {}

    # Subset accuracy (exact match across all labels)
    metrics["subset_accuracy"] = float((y_pred == y_true).all(axis=1).mean())

    # Macro metrics
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average=average, zero_division=0
    )
    metrics["precision"] = float(precision)
    metrics["recall"] = float(recall)
    metrics["f1"] = float(f1)

    # AP / AUC
    try:
        ap = average_precision_score(y_true, y_score, average=average)
        metrics["avg_precision"] = float(ap)
    except ValueError:
        pass

    try:
        auc = roc_auc_score(y_true, y_score, average=average)
        metrics["roc_auc"] = float(auc)
    except ValueError:
        pass

    # Classwise metrics
    per_class = precision_recall_fscore_support(y_true, y_pred, average=None, zero_division=0)
    for label, p, r, f in zip(labels, per_class[0], per_class[1], per_class[2]):
        metrics[f"class/{label}/precision"] = float(p)
        metrics[f"class/{label}/recall"] = float(r)
        metrics[f"class/{label}/f1"] = float(f)

    return metrics


@register_metric("regression")
def regression_metrics(predictions: Any, targets: Any) -> Dict[str, float]:
    """
    Compute regression metrics.

    Args:
        predictions: Predicted values.
        targets: Ground truth values.
    """
    y_true = _to_numpy(targets).reshape(-1)
    y_pred = _to_numpy(predictions).reshape(-1)

    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)

    metrics = {
        "mae": float(mae),
        "mse": float(mse),
        "rmse": float(np.sqrt(mse)),
        "r2": float(r2_score(y_true, y_pred)),
    }
    return metrics


def default_metrics_for_task(task: str) -> list[str]:
    """Return the default metric names for a task type."""
    if task in {"multilabel", "classification", "multi-label"}:
        return ["multilabel_classification"]
    if task in {"regression"}:
        return ["regression"]
    raise ValueError(f"Unknown task type '{task}' for default metrics.")


@dataclass
class MetricCollection:
    """Container to compute multiple metrics in one call."""

    metric_fns: Mapping[str, MetricFn] = field(default_factory=dict)

    @classmethod
    def from_names(cls, names: Iterable[str]) -> MetricCollection:
        missing = [name for name in names if name not in METRIC_REGISTRY]
        if missing:
            raise KeyError(f"Metrics not registered: {missing}")
        return cls({name: METRIC_REGISTRY[name] for name in names})

    def __call__(self, predictions: Any, targets: Any, **kwargs: Any) -> Dict[str, float]:
        results: Dict[str, float] = {}
        for name, fn in self.metric_fns.items():
            metric_values = fn(predictions, targets, **kwargs)
            for key, value in metric_values.items():
                results[f"{name}.{key}"] = float(value)
        return results
