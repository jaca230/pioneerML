from __future__ import annotations

from collections.abc import Mapping

import torch

from .base_classification_evaluator import BaseClassificationEvaluator
from ..factory.registry import REGISTRY as EVALUATOR_REGISTRY


@EVALUATOR_REGISTRY.register("simple_classification")
class SimpleClassificationEvaluator(BaseClassificationEvaluator):
    default_metric_names = ("binary_classification_from_tensors",)
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
        preds_all: list[torch.Tensor] = []
        targets_all: list[torch.Tensor] = []
        threshold = float(config.get("threshold", 0.5))

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
                preds_all.append(preds.detach().cpu())
                targets_all.append(target.detach().cpu())

        if total_samples == 0:
            raise RuntimeError("No samples available for evaluation.")

        preds_cat = torch.cat(preds_all, dim=0)
        targets_cat = torch.cat(targets_all, dim=0)
        loss = total_loss / total_samples

        probs = torch.sigmoid(preds_cat)
        preds_binary = (probs >= float(threshold)).float()
        accuracy = (preds_binary == targets_cat.float()).float().mean().item()
        exact_match = (preds_binary == targets_cat.float()).all(dim=1).float().mean().item()

        target_int = targets_cat.int()
        preds_int = preds_binary.int()
        num_classes = target_int.shape[-1]
        confusion = torch.zeros((num_classes, 2, 2), dtype=torch.int64)
        for cls_idx in range(num_classes):
            truth = target_int[:, cls_idx]
            pred = preds_int[:, cls_idx]
            tn = ((truth == 0) & (pred == 0)).sum().item()
            fp = ((truth == 0) & (pred == 1)).sum().item()
            fn = ((truth == 1) & (pred == 0)).sum().item()
            tp = ((truth == 1) & (pred == 1)).sum().item()
            confusion[cls_idx, 0, 0] += int(tn)
            confusion[cls_idx, 0, 1] += int(fp)
            confusion[cls_idx, 1, 0] += int(fn)
            confusion[cls_idx, 1, 1] += int(tp)

        confusion_metrics = []
        for cls_idx in range(num_classes):
            _, fp = confusion[cls_idx, 0].tolist()
            fn, tp = confusion[cls_idx, 1].tolist()
            total = float(tp + fp + fn)
            confusion_metrics.append({"tp": tp / total, "fp": fp / total, "fn": fn / total} if total > 0 else {"tp": 0.0, "fp": 0.0, "fn": 0.0})

        plot_path = self.resolve_plot_path(dict(config))
        train_history = list(module.train_epoch_loss_history)
        val_history = list(module.val_epoch_loss_history)
        train_total = len(train_history)
        val_total = len(val_history)
        return {
            "metric_context": {
                "preds_binary": preds_binary,
                "targets": targets_cat.float(),
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
