from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Literal, Sequence

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import confusion_matrix

from .base import BasePlot, _overall_normalize, _prepare_classification_inputs

try:
    from IPython.display import display  # type: ignore
except Exception:  # pragma: no cover
    display = None


def _row_normalize(cm: np.ndarray) -> np.ndarray:
    """Normalize confusion matrix by row (each row sums to 1)."""
    row_sums = cm.sum(axis=1, keepdims=True)
    # Avoid division by zero
    row_sums = np.where(row_sums == 0, 1, row_sums)
    return cm.astype(float) / row_sums


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
        normalize_mode: Literal[None, "overall", "row"] | None = None,
        precision: int = 2,
        view_mode: Literal["per_class", "nxn", "summary"] = "per_class",
        summary_mode: Literal["default", "correct_mode", "true_positive_mode"] = "default",
        save_path: str | Path | None = None,
        show: bool = False,
    ) -> str | None:
        # Determine normalization mode
        if normalize_mode is None:
            # Backward compatibility: use normalize boolean
            norm_mode = "overall" if normalize else None
        else:
            norm_mode = normalize_mode
        
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
            
            if view_mode == "summary":
                if summary_mode == "default":
                    # Compact format: 4 columns (TP, TN, FP, FN) x n rows (one per class)
                    summary_data = []
                    summary_data_raw = []
                    
                    for i in range(num_classes):
                        # True Positive: diagonal element (TP for this class)
                        tp = cm_raw[i, i]
                        # False Positive: sum of column i excluding diagonal (predictions of class i that are wrong)
                        fp = cm_raw[:, i].sum() - cm_raw[i, i]
                        # False Negative: sum of row i excluding diagonal (true class i that were predicted as something else)
                        fn = cm_raw[i, :].sum() - cm_raw[i, i]
                        # True Negative: all other correct predictions (total - TP - FP - FN)
                        tn = cm_raw.sum() - tp - fp - fn
                        
                        summary_data_raw.append([tp, tn, fp, fn])
                        
                        if norm_mode == "overall":
                            total = cm_raw.sum()
                            if total > 0:
                                summary_data.append([(tp / total), (tn / total), (fp / total), (fn / total)])
                            else:
                                summary_data.append([0.0, 0.0, 0.0, 0.0])
                        elif norm_mode == "row":
                            row_total = cm_raw[i, :].sum()
                            if row_total > 0:
                                summary_data.append([(tp / row_total), (tn / row_total), (fp / row_total), (fn / row_total)])
                            else:
                                summary_data.append([0.0, 0.0, 0.0, 0.0])
                        else:
                            summary_data.append([tp, tn, fp, fn])
                    
                    summary_matrix = np.array(summary_data)
                    summary_matrix_raw = np.array(summary_data_raw)
                    
                    # Create annotations
                    annot = np.empty_like(summary_matrix, dtype=object)
                    for i in range(summary_matrix.shape[0]):
                        for j in range(summary_matrix.shape[1]):
                            if norm_mode is not None:
                                norm_val = f"{summary_matrix[i, j]:.{precision}f}"
                            else:
                                norm_val = f"{int(summary_matrix_raw[i, j])}"
                            annot[i, j] = f"{norm_val}\n({int(summary_matrix_raw[i, j])})"
                    
                    fig, ax = plt.subplots(figsize=(8, 2 + 0.4 * num_classes))
                    sns.heatmap(
                        summary_matrix,
                        annot=annot,
                        fmt="",
                        cbar=False,
                        xticklabels=["True Positive", "True Negative", "False Positive", "False Negative"],
                        yticklabels=labels,
                        ax=ax,
                    )
                elif summary_mode == "correct_mode":
                    # Correct mode: 3 columns (Correct, False Positive, False Negative) x n rows (one per class)
                    summary_data = []
                    summary_data_raw = []
                    
                    for i in range(num_classes):
                        # True Positive: diagonal element (TP for this class)
                        tp = cm_raw[i, i]
                        # False Positive: sum of column i excluding diagonal (predictions of class i that are wrong)
                        fp = cm_raw[:, i].sum() - cm_raw[i, i]
                        # False Negative: sum of row i excluding diagonal (true class i that were predicted as something else)
                        fn = cm_raw[i, :].sum() - cm_raw[i, i]
                        
                        # For multi-class, "correct" is just TP (diagonal)
                        first_col_value = tp
                        summary_data_raw.append([first_col_value, fp, fn])
                        
                        normalization_total = cm_raw.sum()
                        
                        if norm_mode == "overall":
                            if normalization_total > 0:
                                summary_data.append([(first_col_value / normalization_total), (fp / normalization_total), (fn / normalization_total)])
                            else:
                                summary_data.append([0.0, 0.0, 0.0])
                        elif norm_mode == "row":
                            row_normalization_total = cm_raw[i, :].sum()
                            if row_normalization_total > 0:
                                summary_data.append([(first_col_value / row_normalization_total), (fp / row_normalization_total), (fn / row_normalization_total)])
                            else:
                                summary_data.append([0.0, 0.0, 0.0])
                        else:
                            summary_data.append([first_col_value, fp, fn])
                    
                    summary_matrix = np.array(summary_data)
                    summary_matrix_raw = np.array(summary_data_raw)
                    
                    # Create annotations
                    annot = np.empty_like(summary_matrix, dtype=object)
                    for i in range(summary_matrix.shape[0]):
                        for j in range(summary_matrix.shape[1]):
                            if norm_mode is not None:
                                norm_val = f"{summary_matrix[i, j]:.{precision}f}"
                            else:
                                norm_val = f"{int(summary_matrix_raw[i, j])}"
                            annot[i, j] = f"{norm_val}\n({int(summary_matrix_raw[i, j])})"
                    
                    fig, ax = plt.subplots(figsize=(6, 2 + 0.4 * num_classes))
                    sns.heatmap(
                        summary_matrix,
                        annot=annot,
                        fmt="",
                        cbar=False,
                        xticklabels=["Correct", "False Positive", "False Negative"],
                        yticklabels=labels,
                        ax=ax,
                    )
                else:  # true_positive_mode
                    # True Positive mode: 3 columns (True Positive, False Positive, False Negative) x n rows (one per class)
                    summary_data = []
                    summary_data_raw = []
                    
                    for i in range(num_classes):
                        # True Positive: diagonal element (TP for this class)
                        tp = cm_raw[i, i]
                        # False Positive: sum of column i excluding diagonal (predictions of class i that are wrong)
                        fp = cm_raw[:, i].sum() - cm_raw[i, i]
                        # False Negative: sum of row i excluding diagonal (true class i that were predicted as something else)
                        fn = cm_raw[i, :].sum() - cm_raw[i, i]
                        
                        first_col_value = tp
                        summary_data_raw.append([first_col_value, fp, fn])
                        
                        # Normalize by (TP + FP + FN) excluding TN
                        normalization_total = tp + fp + fn
                        
                        if norm_mode == "overall":
                            if normalization_total > 0:
                                summary_data.append([(first_col_value / normalization_total), (fp / normalization_total), (fn / normalization_total)])
                            else:
                                summary_data.append([0.0, 0.0, 0.0])
                        elif norm_mode == "row":
                            # When using true_positive mode, normalize by (TP + FP + FN) excluding TN
                            row_normalization_total = tp + fp + fn
                            if row_normalization_total > 0:
                                summary_data.append([(first_col_value / row_normalization_total), (fp / row_normalization_total), (fn / row_normalization_total)])
                            else:
                                summary_data.append([0.0, 0.0, 0.0])
                        else:
                            summary_data.append([first_col_value, fp, fn])
                    
                    summary_matrix = np.array(summary_data)
                    summary_matrix_raw = np.array(summary_data_raw)
                    
                    # Create annotations
                    annot = np.empty_like(summary_matrix, dtype=object)
                    for i in range(summary_matrix.shape[0]):
                        for j in range(summary_matrix.shape[1]):
                            if norm_mode is not None:
                                norm_val = f"{summary_matrix[i, j]:.{precision}f}"
                            else:
                                norm_val = f"{int(summary_matrix_raw[i, j])}"
                            annot[i, j] = f"{norm_val}\n({int(summary_matrix_raw[i, j])})"
                    
                    fig, ax = plt.subplots(figsize=(6, 2 + 0.4 * num_classes))
                    sns.heatmap(
                        summary_matrix,
                        annot=annot,
                        fmt="",
                        cbar=False,
                        xticklabels=["True Positive", "False Positive", "False Negative"],
                        yticklabels=labels,
                        ax=ax,
                    )
                ax.set_xlabel("Metric")
                ax.set_ylabel("Class")
                total_samples = cm_raw.sum()
                if norm_mode is not None:
                    ax.set_title(f"Confusion Matrix Summary - Normalized per Class (N={total_samples})")
                else:
                    ax.set_title(f"Confusion Matrix Summary (N={total_samples})")
            else:
                # Standard nxn confusion matrix
                if norm_mode == "overall":
                    cm = _overall_normalize(cm_raw)
                elif norm_mode == "row":
                    cm = _row_normalize(cm_raw)
                else:
                    cm = cm_raw
                
                annot = np.empty_like(cm, dtype=object)
                for i in range(cm.shape[0]):
                    for j in range(cm.shape[1]):
                        if norm_mode is not None:
                            norm_val = f"{cm[i, j]:.{precision}f}"
                        else:
                            norm_val = f"{int(cm_raw[i, j])}"
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
                if norm_mode == "row":
                    ax.set_title(f"Confusion Matrix - Normalized per Class (N={cm_raw.sum()})")
                elif norm_mode == "overall":
                    ax.set_title(f"Confusion Matrix - Normalized Overall (N={cm_raw.sum()})")
                else:
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

        # Multi-label / binary: per-class 2x2 matrices, nxn matrix, or summary view
        y_pred = (y_score >= threshold).astype(int)

        if view_mode == "summary":
            if summary_mode == "default":
                # Compact format: 4 columns (TP, TN, FP, FN) x n rows (one per class)
                summary_data = []
                summary_data_raw = []
                
                for idx in range(num_classes):
                    cm_raw = confusion_matrix(y_true_binary[:, idx], y_pred[:, idx], labels=[0, 1])
                    # cm_raw structure:
                    #        Predicted
                    #        0    1
                    # True 0  TN   FP
                    #     1  FN   TP
                    tn = cm_raw[0, 0]
                    fp = cm_raw[0, 1]
                    fn = cm_raw[1, 0]
                    tp = cm_raw[1, 1]
                    
                    summary_data_raw.append([tp, tn, fp, fn])
                    
                    if norm_mode == "overall":
                        total = cm_raw.sum()
                        if total > 0:
                            summary_data.append([(tp / total), (tn / total), (fp / total), (fn / total)])
                        else:
                            summary_data.append([0.0, 0.0, 0.0, 0.0])
                    elif norm_mode == "row":
                        # For row normalization, normalize by the total samples for this class
                        total = cm_raw.sum()
                        if total > 0:
                            summary_data.append([(tp / total), (tn / total), (fp / total), (fn / total)])
                        else:
                            summary_data.append([0.0, 0.0, 0.0, 0.0])
                    else:
                        summary_data.append([tp, tn, fp, fn])
                
                summary_matrix = np.array(summary_data)
                summary_matrix_raw = np.array(summary_data_raw)
                
                # Create annotations
                annot = np.empty_like(summary_matrix, dtype=object)
                for i in range(summary_matrix.shape[0]):
                    for j in range(summary_matrix.shape[1]):
                        if norm_mode is not None:
                            norm_val = f"{summary_matrix[i, j]:.{precision}f}"
                        else:
                            norm_val = f"{int(summary_matrix_raw[i, j])}"
                        annot[i, j] = f"{norm_val}\n({int(summary_matrix_raw[i, j])})"
                
                fig, ax = plt.subplots(figsize=(8, 2 + 0.4 * num_classes))
                sns.heatmap(
                    summary_matrix,
                    annot=annot,
                    fmt="",
                    cbar=False,
                    xticklabels=["True Positive", "True Negative", "False Positive", "False Negative"],
                    yticklabels=labels,
                    ax=ax,
                )
            elif summary_mode == "correct_mode":
                # Correct mode: 3 columns (Correct, False Positive, False Negative) x n rows (one per class)
                summary_data = []
                summary_data_raw = []
                
                for idx in range(num_classes):
                    cm_raw = confusion_matrix(y_true_binary[:, idx], y_pred[:, idx], labels=[0, 1])
                    # cm_raw structure:
                    #        Predicted
                    #        0    1
                    # True 0  TN   FP
                    #     1  FN   TP
                    tn = cm_raw[0, 0]
                    fp = cm_raw[0, 1]
                    fn = cm_raw[1, 0]
                    tp = cm_raw[1, 1]
                    
                    # Correct = TP + TN
                    first_col_value = tp + tn
                    summary_data_raw.append([first_col_value, fp, fn])
                    
                    # Normalize by total (includes TN)
                    normalization_total = cm_raw.sum()
                    
                    if norm_mode == "overall":
                        if normalization_total > 0:
                            summary_data.append([(first_col_value / normalization_total), (fp / normalization_total), (fn / normalization_total)])
                        else:
                            summary_data.append([0.0, 0.0, 0.0])
                    elif norm_mode == "row":
                        row_total = cm_raw[1, :].sum() if cm_raw.shape[0] > 1 else cm_raw.sum()
                        if row_total > 0:
                            summary_data.append([(first_col_value / row_total), (fp / row_total), (fn / row_total)])
                        else:
                            summary_data.append([0.0, 0.0, 0.0])
                    else:
                        summary_data.append([first_col_value, fp, fn])
                
                summary_matrix = np.array(summary_data)
                summary_matrix_raw = np.array(summary_data_raw)
                
                # Create annotations
                annot = np.empty_like(summary_matrix, dtype=object)
                for i in range(summary_matrix.shape[0]):
                    for j in range(summary_matrix.shape[1]):
                        if norm_mode is not None:
                            norm_val = f"{summary_matrix[i, j]:.{precision}f}"
                        else:
                            norm_val = f"{int(summary_matrix_raw[i, j])}"
                        annot[i, j] = f"{norm_val}\n({int(summary_matrix_raw[i, j])})"
                
                fig, ax = plt.subplots(figsize=(6, 2 + 0.4 * num_classes))
                sns.heatmap(
                    summary_matrix,
                    annot=annot,
                    fmt="",
                    cbar=False,
                    xticklabels=["Correct", "False Positive", "False Negative"],
                    yticklabels=labels,
                    ax=ax,
                )
            else:  # true_positive_mode
                # True Positive mode: 3 columns (True Positive, False Positive, False Negative) x n rows (one per class)
                summary_data = []
                summary_data_raw = []
                
                for idx in range(num_classes):
                    cm_raw = confusion_matrix(y_true_binary[:, idx], y_pred[:, idx], labels=[0, 1])
                    # cm_raw structure:
                    #        Predicted
                    #        0    1
                    # True 0  TN   FP
                    #     1  FN   TP
                    tn = cm_raw[0, 0]
                    fp = cm_raw[0, 1]
                    fn = cm_raw[1, 0]
                    tp = cm_raw[1, 1]
                    
                    # Just TP, drop TN
                    first_col_value = tp
                    summary_data_raw.append([first_col_value, fp, fn])
                    
                    # Normalize by (TP + FP + FN) excluding TN
                    normalization_total = tp + fp + fn
                    
                    if norm_mode == "overall":
                        if normalization_total > 0:
                            summary_data.append([(first_col_value / normalization_total), (fp / normalization_total), (fn / normalization_total)])
                        else:
                            summary_data.append([0.0, 0.0, 0.0])
                    elif norm_mode == "row":
                        # When using true_positive mode, normalize by (TP + FP + FN) excluding TN
                        row_normalization_total = tp + fp + fn
                        if row_normalization_total > 0:
                            summary_data.append([(first_col_value / row_normalization_total), (fp / row_normalization_total), (fn / row_normalization_total)])
                        else:
                            summary_data.append([0.0, 0.0, 0.0])
                    else:
                        summary_data.append([first_col_value, fp, fn])
                
                summary_matrix = np.array(summary_data)
                summary_matrix_raw = np.array(summary_data_raw)
                
                # Create annotations
                annot = np.empty_like(summary_matrix, dtype=object)
                for i in range(summary_matrix.shape[0]):
                    for j in range(summary_matrix.shape[1]):
                        if norm_mode is not None:
                            norm_val = f"{summary_matrix[i, j]:.{precision}f}"
                        else:
                            norm_val = f"{int(summary_matrix_raw[i, j])}"
                        annot[i, j] = f"{norm_val}\n({int(summary_matrix_raw[i, j])})"
                
                fig, ax = plt.subplots(figsize=(6, 2 + 0.4 * num_classes))
                sns.heatmap(
                    summary_matrix,
                    annot=annot,
                    fmt="",
                    cbar=False,
                    xticklabels=["True Positive", "False Positive", "False Negative"],
                    yticklabels=labels,
                    ax=ax,
                )
            ax.set_xlabel("Metric")
            ax.set_ylabel("Class")
            total_samples = y_true_binary.shape[0]
            if norm_mode is not None:
                ax.set_title(f"Confusion Matrix Summary - Normalized per Class (N={total_samples})")
            else:
                ax.set_title(f"Confusion Matrix Summary (N={total_samples})")
        elif view_mode == "nxn":
            # Convert multi-label to single-class predictions using argmax
            # This assumes the model doesn't predict multiple classes simultaneously
            true_mask = y_true_binary.sum(axis=1) > 0
            if not np.any(true_mask):
                raise ValueError("No labeled samples available for confusion matrix.")

            y_true_idx = np.argmax(y_true_binary, axis=1)[true_mask]
            y_pred_idx = np.argmax(y_score, axis=1)[true_mask]
            cm_raw = confusion_matrix(y_true_idx, y_pred_idx, labels=list(range(num_classes)))
            
            if norm_mode == "overall":
                cm = _overall_normalize(cm_raw)
            elif norm_mode == "row":
                cm = _row_normalize(cm_raw)
            else:
                cm = cm_raw
            
            annot = np.empty_like(cm, dtype=object)
            for i in range(cm.shape[0]):
                for j in range(cm.shape[1]):
                    if norm_mode is not None:
                        norm_val = f"{cm[i, j]:.{precision}f}"
                    else:
                        norm_val = f"{int(cm_raw[i, j])}"
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
            if norm_mode == "row":
                ax.set_title(f"Confusion Matrix - Normalized per Class (N={cm_raw.sum()})")
            elif norm_mode == "overall":
                ax.set_title(f"Confusion Matrix - Normalized Overall (N={cm_raw.sum()})")
            else:
                ax.set_title(f"Confusion Matrix (N={cm_raw.sum()})")
        else:
            # Per-class 2x2 matrices (original view)
            cols = min(3, num_classes)
            rows = math.ceil(num_classes / cols)
            fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 4 * rows))
            axes = np.atleast_1d(axes).flatten()

            for idx, ax in enumerate(axes):
                if idx >= num_classes:
                    ax.axis("off")
                    continue
                cm_raw = confusion_matrix(y_true_binary[:, idx], y_pred[:, idx], labels=[0, 1])
                
                if norm_mode == "overall":
                    cm = _overall_normalize(cm_raw)
                elif norm_mode == "row":
                    cm = _row_normalize(cm_raw)
                else:
                    cm = cm_raw
                
                annot = np.empty_like(cm, dtype=object)
                for i in range(cm.shape[0]):
                    for j in range(cm.shape[1]):
                        if norm_mode is not None:
                            norm_val = f"{cm[i, j]:.{precision}f}"
                        else:
                            norm_val = f"{int(cm_raw[i, j])}"
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
                if norm_mode == "row":
                    ax.set_title(f"{labels[idx]} - Normalized per Class (N={cm_raw.sum()})")
                elif norm_mode == "overall":
                    ax.set_title(f"{labels[idx]} - Normalized Overall (N={cm_raw.sum()})")
                else:
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
