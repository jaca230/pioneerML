from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Sequence

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import confusion_matrix

from .base import BasePlot, _overall_normalize, _prepare_classification_inputs

try:
    from IPython.display import display  # type: ignore
except Exception:  # pragma: no cover
    display = None


class ConfusionMatrixPlot(BasePlot):
    name = "multilabel_confusion"

    def render(
        self,
        predictions: Any,
        targets: Any,
        *,
        class_names: Sequence[str] | None = None,
        threshold: float = 0.5,
        normalize: bool = True,
        save_path: str | Path | None = None,
        show: bool = False,
    ) -> str | None:
        y_true_binary, y_score, labels, multi_label, num_classes = _prepare_classification_inputs(
            predictions, targets, class_names
        )

        # Multi-class case: show a single confusion matrix across classes
        if not multi_label and num_classes > 1:
            true_mask = y_true_binary.sum(axis=1) > 0
            if not np.any(true_mask):
                raise ValueError("No labeled samples available for confusion matrix.")

            y_true_idx = np.argmax(y_true_binary, axis=1)[true_mask]
            y_pred_idx = np.argmax(y_score, axis=1)[true_mask]
            cm_raw = confusion_matrix(y_true_idx, y_pred_idx, labels=list(range(num_classes)))
            cm = _overall_normalize(cm_raw) if normalize else cm_raw
            annot = np.empty_like(cm, dtype=object)
            for i in range(cm.shape[0]):
                for j in range(cm.shape[1]):
                    norm_val = f"{cm[i, j]:.2f}" if normalize else f"{int(cm_raw[i, j])}"
                    annot[i, j] = f"{norm_val}\n({int(cm_raw[i, j])})"

            fig, ax = plt.subplots(figsize=(4 + 0.4 * num_classes, 4 + 0.4 * num_classes))
            sns.heatmap(
                cm,
                annot=annot,
                fmt="",
                cbar=False,
                xticklabels=labels,
                yticklabels=labels,
                ax=ax,
            )
            ax.set_xlabel("Predicted")
            ax.set_ylabel("True")
            ax.set_title(f"Confusion Matrix (N={cm_raw.sum()})")

            plt.tight_layout()
            if save_path is not None:
                save_path = str(save_path)
                fig.savefig(save_path, dpi=150, bbox_inches="tight")
            if show:
                backend = plt.get_backend().lower()
                if backend.startswith("agg"):
                    if display is not None:
                        try:
                            display(fig)
                        except Exception:
                            pass
                else:
                    plt.show()
            plt.close(fig)
            return save_path

        # Multi-label / binary: per-class 2x2 matrices
        y_pred = (y_score >= threshold).astype(int)

        cols = min(3, num_classes)
        rows = math.ceil(num_classes / cols)
        fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 4 * rows))
        axes = np.atleast_1d(axes).flatten()

        for idx, ax in enumerate(axes):
            if idx >= num_classes:
                ax.axis("off")
                continue
            cm_raw = confusion_matrix(y_true_binary[:, idx], y_pred[:, idx], labels=[0, 1])
            cm = _overall_normalize(cm_raw) if normalize else cm_raw
            annot = np.empty_like(cm, dtype=object)
            for i in range(cm.shape[0]):
                for j in range(cm.shape[1]):
                    norm_val = f"{cm[i, j]:.2f}" if normalize else f"{int(cm_raw[i, j])}"
                    annot[i, j] = f"{norm_val}\n({int(cm_raw[i, j])})"
            sns.heatmap(
                cm,
                annot=annot,
                fmt="",
                cbar=False,
                ax=ax,
                xticklabels=["0", "1"],
                yticklabels=["0", "1"],
            )
            ax.set_xlabel("Predicted")
            ax.set_ylabel("True")
            ax.set_title(f"{labels[idx]} (N={cm_raw.sum()})")

        plt.tight_layout()
        if save_path is not None:
            save_path = str(save_path)
            fig.savefig(save_path, dpi=150, bbox_inches="tight")
        if show:
            backend = plt.get_backend().lower()
            if backend.startswith("agg"):
                if display is not None:
                    try:
                        display(fig)
                    except Exception:
                        # Fall back silently when display is unavailable
                        pass
            else:
                plt.show()
        plt.close(fig)
        return save_path
