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


def _one_hot_from_int(labels: np.ndarray, num_classes: int) -> np.ndarray:
    labels = labels.astype(int).reshape(-1)
    one_hot = np.zeros((labels.shape[0], num_classes), dtype=int)
    valid = (labels >= 0) & (labels < num_classes)
    one_hot[np.arange(labels.shape[0])[valid], labels[valid]] = 1
    return one_hot


def _pad_columns(arr: np.ndarray, num_classes: int) -> np.ndarray:
    if arr.shape[1] == num_classes:
        return arr
    padded = np.zeros((arr.shape[0], num_classes), dtype=arr.dtype)
    cols = min(arr.shape[1], num_classes)
    padded[:, :cols] = arr[:, :cols]
    return padded


def _prepare_classification_inputs(
    predictions: Any, targets: Any, class_names: Sequence[str] | None
) -> tuple[np.ndarray, np.ndarray, list[str], bool, int]:
    """Normalize predictions/targets to aligned 2D arrays."""
    y_true_raw = _to_numpy(targets)
    y_score_raw = _to_numpy(predictions)

    # Align shapes before normalization so class-index inputs become one-hot.
    if y_true_raw.ndim == 1 and y_score_raw.ndim > 1:
        y_true = _one_hot_from_int(y_true_raw, y_score_raw.shape[1])
    elif y_true_raw.ndim == 1:
        y_true = y_true_raw.reshape(-1, 1)
    else:
        y_true = y_true_raw

    if y_score_raw.ndim == 1 and y_true.ndim > 1:
        y_score = _one_hot_from_int(np.rint(y_score_raw).astype(int), y_true.shape[1])
    elif y_score_raw.ndim == 1:
        y_score = y_score_raw.reshape(-1, 1)
    else:
        y_score = y_score_raw

    num_classes = max(y_true.shape[1], y_score.shape[1])
    y_true = _pad_columns(y_true, num_classes)
    y_score = _pad_columns(y_score, num_classes)

    y_score = _normalize_probs(y_score)
    y_true_binary = (y_true >= 0.5).astype(int)
    multi_label = np.any(y_true_binary.sum(axis=1) > 1)
    labels = _resolve_labels(num_classes, class_names)
    return y_true_binary, y_score, labels, multi_label, num_classes

def _resolve_histories(train_losses, val_losses=None):
    """Accept either explicit loss arrays or a LightningModule with stored histories."""
    if hasattr(train_losses, "train_epoch_loss_history"):
        module = train_losses
        train_losses = (
            getattr(module, "train_epoch_loss_history", None)
            or getattr(module, "train_loss_history", None)
        )
        val_losses = (
            getattr(module, "val_epoch_loss_history", None)
            or getattr(module, "val_loss_history", None)
        )

    train_hist = list(train_losses) if train_losses is not None else []
    val_hist = list(val_losses) if val_losses is not None else []

    # Lightning runs a val sanity check before the first train epoch; trim any
    # leading val entries so lengths align with train epochs.
    while len(val_hist) > len(train_hist) and len(train_hist) > 0:
        val_hist = val_hist[1:]
    return train_hist, val_hist


@register_plot("loss_curves")
def plot_loss_curves(
    train_losses: Iterable[float] | object,
    val_losses: Optional[Iterable[float]] = None,
    *,
    title: str = "Loss Curves",
    xlabel: str = "Epoch",
    save_path: Optional[str] = None,
    show: bool = False,
) -> str | None:
    """
    Plot training/validation loss histories and return the save path (if any),
    for consistency with other evaluation plots.
    """
    train_hist, val_hist = _resolve_histories(train_losses, val_losses)

    fig, ax = plt.subplots(figsize=(6, 4))

    if train_hist:
        ax.plot(train_hist, label="train_loss")
    if val_hist:
        ax.plot(val_hist, label="val_loss")

    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Loss")
    ax.legend()
    fig.tight_layout()

    # Save
    if save_path is not None:
        save_path = str(save_path)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    # Show
    if show:
        backend = plt.get_backend().lower()
        if backend.startswith("agg"):
            try:
                from IPython.display import display
                display(fig)
            except Exception:
                pass
        else:
            plt.show()

    plt.close(fig)
    return save_path


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
    y_true_binary, y_score, labels, multi_label, num_classes = _prepare_classification_inputs(
        predictions, targets, class_names
    )

    if multi_label or num_classes == 1:
        y_pred = (y_score >= threshold).astype(int)
    else:
        pred_idx = np.argmax(y_score, axis=1)
        y_pred = _one_hot_from_int(pred_idx, num_classes)

    cols = min(3, num_classes)
    rows = math.ceil(num_classes / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 4 * rows))
    axes = np.atleast_1d(axes).flatten()

    for idx, ax in enumerate(axes):
        if idx >= num_classes:
            ax.axis("off")
            continue
        cm = confusion_matrix(
            y_true_binary[:, idx], y_pred[:, idx], labels=[0, 1], normalize="all" if normalize else None
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
    y_true_binary, y_score, labels, _, num_classes = _prepare_classification_inputs(
        predictions, targets, class_names
    )

    fig, ax = plt.subplots(figsize=(6, 4))
    for idx in range(num_classes):
        precision, recall, _ = precision_recall_curve(y_true_binary[:, idx], y_score[:, idx])
        ap = average_precision_score(y_true_binary[:, idx], y_score[:, idx])
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
    y_true_binary, y_score, labels, _, num_classes = _prepare_classification_inputs(
        predictions, targets, class_names
    )

    fig, ax = plt.subplots(figsize=(6, 4))
    for idx in range(num_classes):
        fpr, tpr, _ = roc_curve(y_true_binary[:, idx], y_score[:, idx])
        try:
            auc = roc_auc_score(y_true_binary[:, idx], y_score[:, idx])
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
