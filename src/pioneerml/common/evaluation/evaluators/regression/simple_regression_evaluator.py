from __future__ import annotations

from collections.abc import Mapping

import torch

from .base_regression_evaluator import BaseRegressionEvaluator
from ..factory import register_evaluator


@register_evaluator("simple_regression")
class SimpleRegressionEvaluator(BaseRegressionEvaluator):
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
        total_mae = 0.0
        total_samples = 0

        with torch.no_grad():
            for batch in loader:
                batch = batch.to(device)
                raw_preds = module(batch)
                loss, _ = module.compute_loss(raw_preds, batch)
                preds = module.primary_predictions(raw_preds)
                target = module.primary_target(batch, preds)
                mae = torch.abs(preds - target).mean()
                bs = int(target.shape[0])
                total_loss += float(loss.detach().cpu().item()) * bs
                total_mae += float(mae.detach().cpu().item()) * bs
                total_samples += bs

        if total_samples == 0:
            raise RuntimeError("No samples available for evaluation.")

        plot_path = self.resolve_plot_path(dict(config))
        train_history = list(module.train_epoch_loss_history)
        val_history = list(module.val_epoch_loss_history)
        train_total = len(train_history)
        val_total = len(val_history)
        return {
            "plot_kwargs_by_name": {
                "loss_curves": {"train_losses": module, "save_path": plot_path, "show": False},
            },
            "base_metrics": {
                "loss": total_loss / total_samples,
                "mae": total_mae / total_samples,
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
        _ = context
        _ = config
        loss_plot_path = results.get("loss_curves_path")
        if isinstance(loss_plot_path, str):
            results["loss_plot_path"] = loss_plot_path
        return results
