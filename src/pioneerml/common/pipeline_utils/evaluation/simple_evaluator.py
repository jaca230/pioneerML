from __future__ import annotations

from pathlib import Path
from typing import Callable

import torch
from pioneerml.common.evaluation.plots.loss import LossCurvesPlot


class SimpleClassificationEvaluator:
    """Simple shared evaluation logic for graph classification-style tasks."""

    def _concise_history(self, history: list[float], *, max_points: int = 20) -> tuple[list[float], int]:
        if len(history) <= max_points:
            return list(history), len(history)
        return list(history[-max_points:]), len(history)

    def resolve_plot_path(self, config: dict | None) -> str | None:
        if not config:
            return None
        if config.get("plot_path"):
            return str(config["plot_path"])
        if config.get("plot_dir"):
            plot_dir = Path(str(config["plot_dir"]))
            plot_dir.mkdir(parents=True, exist_ok=True)
            return str(plot_dir / "loss_curves.png")
        return None

    def evaluate(
        self,
        *,
        module,
        graphs: list,
        threshold: float,
        batch_size: int,
        loader_cls,
        collate_fn: Callable | None = None,
        plot_config: dict | None = None,
    ) -> dict:
        if not graphs:
            raise RuntimeError("No graphs available for evaluation.")

        module.eval()

        loader_kwargs = {"batch_size": int(batch_size), "shuffle": False}
        if collate_fn is not None:
            loader_kwargs["collate_fn"] = collate_fn
        loader = loader_cls(graphs, **loader_kwargs)

        device = next(module.parameters()).device
        total_loss = 0.0
        total_samples = 0
        preds_all = []
        targets_all = []

        with torch.no_grad():
            for batch in loader:
                batch = batch.to(device)
                logits = module(batch)
                preds = logits[0] if isinstance(logits, (tuple, list)) else logits
                target = batch.y
                if target.dim() == 1 and preds.dim() == 2 and target.numel() % preds.shape[-1] == 0:
                    target = target.view(-1, preds.shape[-1])
                loss = module.loss_fn(preds, target)
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
            if total > 0:
                confusion_metrics.append({"tp": tp / total, "fp": fp / total, "fn": fn / total})
            else:
                confusion_metrics.append({"tp": 0.0, "fp": 0.0, "fn": 0.0})

        plot_path = self.resolve_plot_path(plot_config)
        if plot_path is not None:
            LossCurvesPlot().render(module, save_path=plot_path, show=False)

        train_history_raw = list(module.train_epoch_loss_history)
        val_history_raw = list(module.val_epoch_loss_history)
        train_history, train_total = self._concise_history(train_history_raw)
        val_history, val_total = self._concise_history(val_history_raw)

        return {
            "loss": float(loss),
            "accuracy": float(accuracy),
            "exact_match": float(exact_match),
            "confusion": confusion_metrics,
            "threshold": float(threshold),
            "train_loss_history": train_history,
            "train_loss_history_total_points": train_total,
            "val_loss_history": val_history,
            "val_loss_history_total_points": val_total,
            "loss_plot_path": plot_path,
        }
