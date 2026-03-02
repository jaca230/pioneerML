from __future__ import annotations

from typing import Any

from ..base_classification_metric import BaseClassificationMetric


class BaseBinaryClassificationMetric(BaseClassificationMetric):
    """Base helpers for binary multi-label classification metrics."""

    @staticmethod
    def none_metrics() -> dict[str, Any]:
        return {
            "accuracy": None,
            "exact_match": None,
            "confusion": None,
        }

    @staticmethod
    def normalized_confusion(*, tn: list[int], fp: list[int], fn: list[int], tp: list[int]) -> list[dict[str, float]]:
        confusion: list[dict[str, float]] = []
        for i in range(min(len(tn), len(tp), len(fp), len(fn))):
            total = float(tn[i] + tp[i] + fp[i] + fn[i])
            if total > 0:
                confusion.append(
                    {
                        "tn": tn[i] / total,
                        "fp": fp[i] / total,
                        "fn": fn[i] / total,
                        "tp": tp[i] / total,
                    }
                )
            else:
                confusion.append({"tn": 0.0, "fp": 0.0, "fn": 0.0, "tp": 0.0})
        return confusion
