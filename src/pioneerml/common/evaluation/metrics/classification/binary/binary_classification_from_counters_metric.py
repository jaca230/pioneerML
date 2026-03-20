from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from ...registry import REGISTRY as METRIC_REGISTRY
from .base_binary_classification_metric import BaseBinaryClassificationMetric


@METRIC_REGISTRY.register("binary_classification_from_counters")
class BinaryClassificationFromCountersMetric(BaseBinaryClassificationMetric):
    """Computes accuracy/exact/confusion from aggregated counters."""

    def compute(self, *, context: Mapping[str, Any]) -> dict[str, Any]:
        counters = context.get("counters")
        if not isinstance(counters, Mapping) or not bool(counters.get("has_targets", False)):
            return self.none_metrics()

        label_total = int(counters.get("label_total", 0))
        label_equal = int(counters.get("label_equal", 0))
        graph_total = int(counters.get("graph_total", 0))
        graph_exact = int(counters.get("graph_exact", 0))
        tn = [int(v) for v in counters.get("tn", [])]
        tp = [int(v) for v in counters.get("tp", [])]
        fp = [int(v) for v in counters.get("fp", [])]
        fn = [int(v) for v in counters.get("fn", [])]

        return {
            "accuracy": (float(label_equal) / float(label_total)) if label_total > 0 else None,
            "exact_match": (float(graph_exact) / float(graph_total)) if graph_total > 0 else None,
            "confusion": self.normalized_confusion(tn=tn, tp=tp, fp=fp, fn=fn),
        }
