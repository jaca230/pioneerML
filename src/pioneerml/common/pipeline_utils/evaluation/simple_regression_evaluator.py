from __future__ import annotations

from pathlib import Path
from typing import Callable

import torch
from pioneerml.common.evaluation.plots.loss import LossCurvesPlot


class SimpleRegressionEvaluator:
    """Simple shared evaluation logic for graph regression tasks."""

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
        total_mae = 0.0
        total_samples = 0

        with torch.no_grad():
            for batch in loader:
                batch = batch.to(device)
                preds = module(batch)
                preds = preds[0] if isinstance(preds, (tuple, list)) else preds
                target = batch.y
                if target.dim() == 1 and preds.dim() == 2 and target.numel() % preds.shape[-1] == 0:
                    target = target.view(-1, preds.shape[-1])
                loss = module.loss_fn(preds, target)
                mae = torch.abs(preds - target).mean()
                bs = int(target.shape[0])
                total_loss += float(loss.detach().cpu().item()) * bs
                total_mae += float(mae.detach().cpu().item()) * bs
                total_samples += bs

        if total_samples == 0:
            raise RuntimeError("No samples available for evaluation.")

        plot_path = self.resolve_plot_path(plot_config)
        if plot_path is not None:
            LossCurvesPlot().render(module, save_path=plot_path, show=False)

        train_history_raw = list(module.train_epoch_loss_history)
        val_history_raw = list(module.val_epoch_loss_history)
        train_history, train_total = self._concise_history(train_history_raw)
        val_history, val_total = self._concise_history(val_history_raw)

        return {
            "loss": total_loss / total_samples,
            "mae": total_mae / total_samples,
            "train_loss_history": train_history,
            "train_loss_history_total_points": train_total,
            "val_loss_history": val_history,
            "val_loss_history_total_points": val_total,
            "loss_plot_path": plot_path,
        }
