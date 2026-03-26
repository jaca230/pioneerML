from __future__ import annotations

from typing import Any, Sequence

import numpy as np

from ..base_plot import BasePlot, _to_numpy


class ClassificationPlotBase(BasePlot):
    """Base class for classification plots with normalized classifier inputs."""

    def prepare_classification_inputs(
        self,
        *,
        predictions: Any,
        targets: Any,
        class_names: Sequence[str] | None = None,
    ) -> tuple[np.ndarray, np.ndarray, list[str], bool, int]:
        y_true_raw = _to_numpy(targets)
        y_score_raw = _to_numpy(predictions)

        if y_true_raw.ndim == 1 and y_score_raw.ndim > 1:
            if y_true_raw.size % y_score_raw.shape[1] == 0:
                y_true = y_true_raw.reshape(-1, y_score_raw.shape[1])
            else:
                y_true = self.one_hot_from_int(y_true_raw, y_score_raw.shape[1])
        elif y_true_raw.ndim == 1:
            y_true = y_true_raw.reshape(-1, 1)
        else:
            y_true = y_true_raw

        if y_score_raw.ndim == 1 and y_true.ndim > 1:
            y_score = self.one_hot_from_int(np.rint(y_score_raw).astype(int), y_true.shape[1])
        elif y_score_raw.ndim == 1:
            y_score = y_score_raw.reshape(-1, 1)
        else:
            y_score = y_score_raw

        num_classes = max(y_true.shape[1], y_score.shape[1])
        y_true = self.pad_columns(y_true, num_classes)
        y_score = self.pad_columns(y_score, num_classes)

        y_score = self.normalize_probs(y_score)
        y_true_binary = (y_true >= 0.5).astype(int)
        multi_label = np.any(y_true_binary.sum(axis=1) > 1)
        labels = self.resolve_labels(num_classes, class_names)
        return y_true_binary, y_score, labels, multi_label, num_classes

    @staticmethod
    def overall_normalize(cm):
        total = cm.sum()
        if total == 0:
            return cm.astype(float)
        return cm.astype(float) / float(total)

    @staticmethod
    def normalize_probs(scores: np.ndarray) -> np.ndarray:
        if scores.min() < 0.0 or scores.max() > 1.0:
            scores = np.clip(scores, -60.0, 60.0)
            return 1.0 / (1.0 + np.exp(-scores))
        return scores

    @staticmethod
    def resolve_labels(num_classes: int, class_names: Sequence[str] | None) -> list[str]:
        labels = list(class_names) if class_names is not None else [str(i) for i in range(num_classes)]
        if len(labels) < num_classes:
            labels.extend(str(i) for i in range(len(labels), num_classes))
        return labels[:num_classes]

    @staticmethod
    def one_hot_from_int(labels: np.ndarray, num_classes: int) -> np.ndarray:
        labels = labels.astype(int).reshape(-1)
        one_hot = np.zeros((labels.shape[0], num_classes), dtype=int)
        valid = (labels >= 0) & (labels < num_classes)
        one_hot[np.arange(labels.shape[0])[valid], labels[valid]] = 1
        return one_hot

    @staticmethod
    def pad_columns(arr: np.ndarray, num_classes: int) -> np.ndarray:
        if arr.shape[1] == num_classes:
            return arr
        padded = np.zeros((arr.shape[0], num_classes), dtype=arr.dtype)
        cols = min(arr.shape[1], num_classes)
        padded[:, :cols] = arr[:, :cols]
        return padded
