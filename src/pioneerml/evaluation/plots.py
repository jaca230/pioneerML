"""
Standardized evaluation plots for diagnostics.

Plot registry makes it easy for collaborators to add new visualizations without
touching core code paths.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Callable, Sequence

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)

PLOT_REGISTRY: dict[str, Callable[..., str | None]] = {}


def register_plot(name: str) -> Callable[[Callable[..., str | None]], Callable[..., str | None]]:
    """Decorator to register a plot function by name."""

    def decorator(fn: Callable[..., str | None]) -> Callable[..., str | None]:
        PLOT_REGISTRY[name] = fn
        return fn

    return decorator


def _to_numpy(arr: Any) -> np.ndarray:
    if torch.is_tensor(arr):
        return arr.detach().cpu().numpy()
    if isinstance(arr, (list, tuple)):
        return np.asarray(arr)
    if isinstance(arr, np.ndarray):
        return arr
    raise TypeError(f"Unsupported input type for plotting: {type(arr)}")


def _normalize_probs(scores: np.ndarray) -> np.ndarray:
    if scores.min() < 0.0 or scores.max() > 1.0:
        return 1.0 / (1.0 + np.exp(-scores))
    return scores


def _resolve_labels(num_classes: int, class_names: Sequence[str] | None) -> list[str]:
    labels = list(class_names) if class_names is not None else [str(i) for i in range(num_classes)]
    if len(labels) < num_classes:
        labels.extend(str(i) for i in range(len(labels), num_classes))
    return labels[:num_classes]


@register_plot("multilabel_confusion")
def plot_multilabel_confusion_matrix(
    predictions: Any,
    targets: Any,
    *,
    class_names: Sequence[str] | None = None,
    threshold: float = 0.5,
    normalize: bool = True,
    save_path: str | Path | None = None,
    show: bool = False,
) -> str | None:
    """
    Plot per-class confusion matrices for multi-label outputs.
    """
    y_true = _to_numpy(targets)
    y_score = _normalize_probs(_to_numpy(predictions))

    if y_true.ndim == 1:
        y_true = y_true.reshape(-1, 1)
    if y_score.ndim == 1:
        y_score = y_score.reshape(-1, 1)

    # Normalize targets to predictions shape if possible
    if y_true.size == y_score.size and y_true.shape != y_score.shape:
        y_true = y_true.reshape(y_score.shape)

    y_pred = (y_score >= threshold).astype(int)
    num_classes = y_true.shape[1]
    labels = _resolve_labels(num_classes, class_names)

    cols = min(3, num_classes)
    rows = math.ceil(num_classes / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 4 * rows))
    axes = np.atleast_1d(axes).flatten()

    for idx, ax in enumerate(axes):
        if idx >= num_classes:
            ax.axis("off")
            continue
        cm = confusion_matrix(
            y_true[:, idx], y_pred[:, idx], labels=[0, 1], normalize="all" if normalize else None
        )
        if normalize and cm.sum() > 0:
            cm = cm / cm.sum()
        sns.heatmap(
            cm,
            annot=True,
            fmt=".2f" if normalize else "d",
            cbar=False,
            ax=ax,
            xticklabels=["0", "1"],
            yticklabels=["0", "1"],
        )
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        ax.set_title(labels[idx])

    plt.tight_layout()
    if save_path is not None:
        save_path = str(save_path)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    if show:
        backend = plt.get_backend().lower()
        if backend.startswith("agg"):
            try:
                from IPython.display import display
                display(fig)
            except Exception:
                # Fall back silently when display is unavailable
                pass
        else:
            plt.show()
    plt.close(fig)
    return save_path


@register_plot("precision_recall")
def plot_precision_recall_curves(
    predictions: Any,
    targets: Any,
    *,
    class_names: Sequence[str] | None = None,
    save_path: str | Path | None = None,
    show: bool = False,
) -> str | None:
    """
    Plot per-class precision-recall curves with Average Precision labels.
    """
    y_true = _to_numpy(targets)
    y_score = _normalize_probs(_to_numpy(predictions))

    if y_true.ndim == 1:
        y_true = y_true.reshape(-1, 1)
    if y_score.ndim == 1:
        y_score = y_score.reshape(-1, 1)

    # Normalize targets to predictions shape if possible
    if y_true.size == y_score.size and y_true.shape != y_score.shape:
        y_true = y_true.reshape(y_score.shape)

    num_classes = y_true.shape[1]
    labels = _resolve_labels(num_classes, class_names)

    fig, ax = plt.subplots(figsize=(6, 4))
    for idx in range(num_classes):
        precision, recall, _ = precision_recall_curve(y_true[:, idx], y_score[:, idx])
        ap = average_precision_score(y_true[:, idx], y_score[:, idx])
        ax.plot(recall, precision, label=f"{labels[idx]} (AP={ap:.3f})")

    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall Curves")
    ax.legend()
    ax.grid(True, linestyle="--", alpha=0.4)
    plt.tight_layout()

    if save_path is not None:
        save_path = str(save_path)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    if show:
        backend = plt.get_backend().lower()
        if backend.startswith("agg"):
            try:
                from IPython.display import display
                display(fig)
            except Exception:
                # Fall back silently when display is unavailable
                pass
        else:
            plt.show()
    plt.close(fig)
    return save_path


@register_plot("roc")
def plot_roc_curves(
    predictions: Any,
    targets: Any,
    *,
    class_names: Sequence[str] | None = None,
    save_path: str | Path | None = None,
    show: bool = False,
) -> str | None:
    """
    Plot per-class ROC curves with AUC labels.
    """
    y_true = _to_numpy(targets)
    y_score = _normalize_probs(_to_numpy(predictions))

    if y_true.ndim == 1:
        y_true = y_true.reshape(-1, 1)
    if y_score.ndim == 1:
        y_score = y_score.reshape(-1, 1)

    # Normalize targets to predictions shape if possible
    if y_true.size == y_score.size and y_true.shape != y_score.shape:
        y_true = y_true.reshape(y_score.shape)

    num_classes = y_true.shape[1]
    labels = _resolve_labels(num_classes, class_names)

    fig, ax = plt.subplots(figsize=(6, 4))
    for idx in range(num_classes):
        fpr, tpr, _ = roc_curve(y_true[:, idx], y_score[:, idx])
        try:
            auc = roc_auc_score(y_true[:, idx], y_score[:, idx])
            label = f"{labels[idx]} (AUC={auc:.3f})"
        except ValueError:
            label = labels[idx]
        ax.plot(fpr, tpr, label=label)

    ax.plot([0, 1], [0, 1], "k--", alpha=0.6, label="Chance")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curves")
    ax.legend()
    ax.grid(True, linestyle="--", alpha=0.4)
    plt.tight_layout()

    if save_path is not None:
        save_path = str(save_path)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    if show:
        backend = plt.get_backend().lower()
        if backend.startswith("agg"):
            try:
                from IPython.display import display
                display(fig)
            except Exception:
                # Fall back silently when display is unavailable
                pass
        else:
            plt.show()
    plt.close(fig)
    return save_path


@register_plot("regression_diagnostics")
def plot_regression_diagnostics(
    predictions: Any,
    targets: Any,
    *,
    save_path: str | Path | None = None,
    show: bool = False,
) -> str | None:
    """
    Plot regression residuals and prediction vs target scatter.
    """
    y_true = _to_numpy(targets).reshape(-1)
    y_pred = _to_numpy(predictions).reshape(-1)
    residuals = y_pred - y_true

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    # Scatter plot
    axes[0].scatter(y_true, y_pred, alpha=0.6)
    min_val = min(y_true.min(), y_pred.min())
    max_val = max(y_true.max(), y_pred.max())
    axes[0].plot([min_val, max_val], [min_val, max_val], "k--", alpha=0.6)
    axes[0].set_xlabel("True")
    axes[0].set_ylabel("Predicted")
    axes[0].set_title("Predicted vs True")
    axes[0].grid(True, linestyle="--", alpha=0.4)

    # Residuals
    sns.histplot(residuals, kde=True, ax=axes[1])
    axes[1].axvline(0, color="k", linestyle="--", alpha=0.7)
    axes[1].set_xlabel("Residual")
    axes[1].set_title("Residual Distribution")

    plt.tight_layout()
    if save_path is not None:
        save_path = str(save_path)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    if show:
        backend = plt.get_backend().lower()
        if backend.startswith("agg"):
            try:
                from IPython.display import display
                display(fig)
            except Exception:
                # Fall back silently when display is unavailable
                pass
        else:
            plt.show()
    plt.close(fig)
    return save_path
