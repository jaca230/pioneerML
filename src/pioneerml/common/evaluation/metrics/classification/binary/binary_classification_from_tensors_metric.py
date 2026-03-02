from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch

from ...registry import register_step_metric
from .base_binary_classification_metric import BaseBinaryClassificationMetric


@register_step_metric("binary_classification_from_tensors")
class BinaryClassificationFromTensorsMetric(BaseBinaryClassificationMetric):
    """Computes accuracy/exact/confusion from prediction and target tensors."""

    def compute(self, *, context: Mapping[str, Any]) -> dict[str, Any]:
        preds_binary = context.get("preds_binary")
        targets = context.get("targets")
        if preds_binary is None or targets is None:
            return self.none_metrics()
        if not torch.is_tensor(preds_binary) or not torch.is_tensor(targets):
            return self.none_metrics()
        if targets.numel() == 0:
            return self.none_metrics()

        preds = preds_binary.to(torch.float32)
        truth = targets.to(torch.float32)
        if preds.ndim == 1:
            preds = preds.unsqueeze(1)
        if truth.ndim == 1:
            truth = truth.unsqueeze(1)

        tn: list[int] = []
        tp: list[int] = []
        fp: list[int] = []
        fn: list[int] = []
        for cls_idx in range(int(truth.shape[1])):
            cls_truth = truth[:, cls_idx].to(torch.int64)
            cls_pred = preds[:, cls_idx].to(torch.int64)
            tn.append(int(((cls_truth == 0) & (cls_pred == 0)).sum().item()))
            fp.append(int(((cls_truth == 0) & (cls_pred == 1)).sum().item()))
            fn.append(int(((cls_truth == 1) & (cls_pred == 0)).sum().item()))
            tp.append(int(((cls_truth == 1) & (cls_pred == 1)).sum().item()))

        return {
            "accuracy": float((preds == truth).to(torch.float32).mean().item()),
            "exact_match": float(((preds == truth).all(dim=1)).to(torch.float32).mean().item()),
            "confusion": self.normalized_confusion(tn=tn, tp=tp, fp=fp, fn=fn),
        }
