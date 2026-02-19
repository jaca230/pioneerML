from __future__ import annotations

import torch

from pioneerml.common.evaluation.evaluators import SimpleRegressionEvaluator
from pioneerml.common.pipeline.services import BaseEvaluationService

from ..base import PositronAngleServiceBase


class PositronAngleEvaluationService(PositronAngleServiceBase, BaseEvaluationService):
    step_key = "evaluate"

    def __init__(self, *, dataset, module, pipeline_config: dict | None = None) -> None:
        super().__init__(dataset=dataset, pipeline_config=pipeline_config)
        self.module = module
        self.evaluator = SimpleRegressionEvaluator()

    def default_config(self) -> dict:
        return {
            "batch_size": 64,
            "chunk_row_groups": 4,
            "chunk_workers": 0,
            "use_group_probs": True,
            "use_splitter_probs": True,
            "use_endpoint_preds": True,
            "use_event_splitter_affinity": True,
            "training_relevant_only": True,
        }

    def _evaluate_from_loader(self, *, loader, plot_config: dict | None) -> dict:
        self.module.eval()
        device = next(self.module.parameters()).device
        total_loss = 0.0
        total_mae = 0.0
        total_samples = 0

        with torch.no_grad():
            for batch in loader:
                batch = batch.to(device, non_blocking=True)
                preds = self.module(batch)
                preds = preds[0] if isinstance(preds, (tuple, list)) else preds
                target = batch.y
                if target.dim() == 1 and preds.dim() == 2 and target.numel() % preds.shape[-1] == 0:
                    target = target.view(-1, preds.shape[-1])
                loss = self.module.loss_fn(preds, target)
                mae = torch.abs(preds - target).mean()
                bs = int(target.shape[0])
                total_loss += float(loss.detach().cpu().item()) * bs
                total_mae += float(mae.detach().cpu().item()) * bs
                total_samples += bs

        if total_samples == 0:
            raise RuntimeError("No samples available for evaluation.")

        plot_path = self.evaluator.resolve_plot_path(plot_config)
        if plot_path is not None:
            from pioneerml.common.evaluation.plots.loss import LossCurvesPlot

            LossCurvesPlot().render(self.module, save_path=plot_path, show=False)
        train_history, train_total = self.evaluator.concise_history(list(self.module.train_epoch_loss_history))
        val_history, val_total = self.evaluator.concise_history(list(self.module.val_epoch_loss_history))
        return {
            "loss": total_loss / total_samples,
            "mae": total_mae / total_samples,
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
            raise RuntimeError("PositronAngleGraphLoader must run in train mode for evaluation.")
        loader = provider.make_dataloader(shuffle_batches=False)
        return self._evaluate_from_loader(loader=loader, plot_config=cfg)
