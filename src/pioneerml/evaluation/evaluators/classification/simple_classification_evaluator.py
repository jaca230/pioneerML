from __future__ import annotations

from collections.abc import Mapping

import torch

from .base_classification_evaluator import BaseClassificationEvaluator
from ..factory.registry import REGISTRY as EVALUATOR_REGISTRY


@EVALUATOR_REGISTRY.register("simple_classification")
class SimpleClassificationEvaluator(BaseClassificationEvaluator):
    default_metric_names = ("binary_classification_from_counters",)
    default_plot_names = ("loss_curves",)

    def build_context(
        self,
        *,
        module,
        loader,
        config: Mapping[str, object],
    ) -> dict[str, object]:
        module.eval()

        device = next(module.parameters()).device
        total_loss = 0.0
        total_samples = 0
        threshold = float(config.get("threshold", 0.5))
        label_total = 0
        label_equal = 0
        graph_total = 0
        graph_exact = 0
        num_classes: int | None = None
        tn: list[int] = []
        fp: list[int] = []
        fn: list[int] = []
        tp: list[int] = []

        with torch.no_grad():
            for batch in loader:
                batch = batch.to(device, non_blocking=True)
                logits = module(batch)
                loss, _ = module.compute_loss(logits, batch)
                preds = module.primary_predictions(logits)
                target = module.primary_target(batch, preds)
                bs = int(target.shape[0])
                total_loss += loss.detach().cpu().item() * bs
                total_samples += bs

                probs = torch.sigmoid(preds.detach())
                preds_binary = (probs >= float(threshold)).to(torch.int64)
                targets = target.detach().to(torch.int64)
                if preds_binary.ndim == 1:
                    preds_binary = preds_binary.unsqueeze(1)
                if targets.ndim == 1:
                    targets = targets.unsqueeze(1)
                if preds_binary.shape != targets.shape:
                    raise RuntimeError(
                        "simple_classification evaluator requires predictions and targets with matching shape. "
                        f"Got {tuple(preds_binary.shape)} vs {tuple(targets.shape)}."
                    )

                current_num_classes = int(targets.shape[1])
                if num_classes is None:
                    num_classes = current_num_classes
                    tn = [0 for _ in range(num_classes)]
                    fp = [0 for _ in range(num_classes)]
                    fn = [0 for _ in range(num_classes)]
                    tp = [0 for _ in range(num_classes)]
                elif int(num_classes) != int(current_num_classes):
                    raise RuntimeError(
                        "simple_classification evaluator saw varying class dimensions across batches. "
                        f"Expected {num_classes}, got {current_num_classes}."
                    )

                equal = preds_binary == targets
                label_total += int(equal.numel())
                label_equal += int(equal.sum().item())
                graph_total += int(equal.shape[0])
                graph_exact += int(equal.all(dim=1).sum().item())

                for cls_idx in range(current_num_classes):
                    truth = targets[:, cls_idx]
                    pred = preds_binary[:, cls_idx]
                    tn[cls_idx] += int(((truth == 0) & (pred == 0)).sum().item())
                    fp[cls_idx] += int(((truth == 0) & (pred == 1)).sum().item())
                    fn[cls_idx] += int(((truth == 1) & (pred == 0)).sum().item())
                    tp[cls_idx] += int(((truth == 1) & (pred == 1)).sum().item())

        if total_samples == 0:
            raise RuntimeError("No samples available for evaluation.")
        if num_classes is None:
            raise RuntimeError("No class targets were observed during evaluation.")

        loss = total_loss / total_samples
        accuracy = (float(label_equal) / float(label_total)) if label_total > 0 else 0.0
        exact_match = (float(graph_exact) / float(graph_total)) if graph_total > 0 else 0.0
        confusion_metrics = []
        for cls_idx in range(num_classes):
            total = float(tp[cls_idx] + fp[cls_idx] + fn[cls_idx])
            confusion_metrics.append(
                {
                    "tp": float(tp[cls_idx]) / total,
                    "fp": float(fp[cls_idx]) / total,
                    "fn": float(fn[cls_idx]) / total,
                }
                if total > 0
                else {"tp": 0.0, "fp": 0.0, "fn": 0.0}
            )

        plot_path = self.resolve_plot_path(dict(config))
        train_history = list(module.train_epoch_loss_history)
        val_history = list(module.val_epoch_loss_history)
        train_total = len(train_history)
        val_total = len(val_history)
        return {
            "metric_context": {
                "counters": {
                    "has_targets": True,
                    "label_total": int(label_total),
                    "label_equal": int(label_equal),
                    "graph_total": int(graph_total),
                    "graph_exact": int(graph_exact),
                    "tn": [int(v) for v in tn],
                    "tp": [int(v) for v in tp],
                    "fp": [int(v) for v in fp],
                    "fn": [int(v) for v in fn],
                }
            },
            "plot_kwargs_by_name": {
                "loss_curves": {"train_losses": module, "save_path": plot_path, "show": False},
            },
            "base_metrics": {
                "loss": float(loss),
                "accuracy": float(accuracy),
                "exact_match": float(exact_match),
                "confusion": confusion_metrics,
                "threshold": float(threshold),
                "train_loss_history": train_history,
                "train_loss_history_total_points": train_total,
                "val_loss_history": val_history,
                "val_loss_history_total_points": val_total,
            },
        }

    def finalize_results(
        self,
        *,
        results: dict[str, object],
        context: Mapping[str, object],
        config: Mapping[str, object],
    ) -> dict[str, object]:
        _ = config
        loss_plot_path = results.get("loss_curves_path")
        if isinstance(loss_plot_path, str):
            results["loss_plot_path"] = loss_plot_path
        return results
