from __future__ import annotations

import torch

from pioneerml.common.evaluation.evaluators import SimpleClassificationEvaluator
from pioneerml.common.pipeline.services import BaseEvaluationService

from ..base import GroupSplitterServiceBase


class GroupSplitterEvaluationService(GroupSplitterServiceBase, BaseEvaluationService):
    step_key = "evaluate"

    def __init__(self, *, dataset, module, pipeline_config: dict | None = None) -> None:
        super().__init__(dataset=dataset, pipeline_config=pipeline_config)
        self.module = module
        self.evaluator = SimpleClassificationEvaluator()

    def default_config(self) -> dict:
        return {
            "threshold": 0.5,
            "batch_size": 64,
            "chunk_row_groups": 4,
            "chunk_workers": 0,
            "use_group_probs": True,
        }

    def _evaluate_from_loader(self, *, loader, threshold: float, plot_config: dict | None) -> dict:
        self.module.eval()
        device = next(self.module.parameters()).device
        total_loss = 0.0
        total_samples = 0
        preds_all: list[torch.Tensor] = []
        targets_all: list[torch.Tensor] = []

        with torch.no_grad():
            for batch in loader:
                batch = batch.to(device, non_blocking=True)
                logits = self.module(batch)
                preds = logits[0] if isinstance(logits, (tuple, list)) else logits
                target = batch.y
                if target.dim() == 1 and preds.dim() == 2 and target.numel() % preds.shape[-1] == 0:
                    target = target.view(-1, preds.shape[-1])
                loss = self.module.loss_fn(preds, target)
                bs = int(target.shape[0])
                total_loss += float(loss.detach().cpu().item()) * bs
                total_samples += bs
                preds_all.append(preds.detach().cpu())
                targets_all.append(target.detach().cpu())

        if total_samples == 0:
            raise RuntimeError("No samples available for evaluation.")

        preds_cat = torch.cat(preds_all, dim=0)
        targets_cat = torch.cat(targets_all, dim=0)
        probs = torch.sigmoid(preds_cat)
        preds_binary = (probs >= float(threshold)).float()
        target_float = targets_cat.float()
        accuracy = float((preds_binary == target_float).float().mean().item())
        exact_match = float((preds_binary == target_float).all(dim=1).float().mean().item())
        loss = float(total_loss / total_samples)

        target_int = targets_cat.int()
        preds_int = preds_binary.int()
        num_classes = target_int.shape[-1]
        confusion_metrics = []
        for cls_idx in range(num_classes):
            truth = target_int[:, cls_idx]
            pred = preds_int[:, cls_idx]
            fp = int(((truth == 0) & (pred == 1)).sum().item())
            fn = int(((truth == 1) & (pred == 0)).sum().item())
            tp = int(((truth == 1) & (pred == 1)).sum().item())
            total = float(tp + fp + fn)
            confusion_metrics.append({"tp": tp / total, "fp": fp / total, "fn": fn / total} if total > 0 else {"tp": 0.0, "fp": 0.0, "fn": 0.0})

        plot_path = self.evaluator.resolve_plot_path(plot_config)
        if plot_path is not None:
            from pioneerml.common.evaluation.plots.loss import LossCurvesPlot

            LossCurvesPlot().render(self.module, save_path=plot_path, show=False)
        train_history, train_total = self.evaluator.concise_history(list(self.module.train_epoch_loss_history))
        val_history, val_total = self.evaluator.concise_history(list(self.module.val_epoch_loss_history))
        return {
            "loss": loss,
            "accuracy": accuracy,
            "exact_match": exact_match,
            "confusion": confusion_metrics,
            "threshold": float(threshold),
            "train_loss_history": train_history,
            "train_loss_history_total_points": train_total,
            "val_loss_history": val_history,
            "val_loss_history_total_points": val_total,
            "loss_plot_path": plot_path,
        }

    def execute(self) -> dict:
        cfg = self.get_config()
        params = self._resolve_loader_params(cfg, purpose="evaluate")
        raw_loader_cfg = cfg.get("loader_config")
        if isinstance(raw_loader_cfg, dict):
            if not isinstance(raw_loader_cfg.get("evaluate"), dict) and isinstance(raw_loader_cfg.get("val"), dict):
                params = self._resolve_loader_params(cfg, purpose="val")
        provider = self.loader_factory.build_loader(loader_params=params)
        if not provider.include_targets:
            raise RuntimeError("GroupSplitterGraphLoader must run in train mode for evaluation.")
        loader = provider.make_dataloader(shuffle_batches=False)
        return self._evaluate_from_loader(loader=loader, threshold=float(cfg.get("threshold", 0.5)), plot_config=cfg)
