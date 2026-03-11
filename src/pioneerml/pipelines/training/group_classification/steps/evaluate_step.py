
from typing import Any

import torch
from zenml import step

from pioneerml.common.evaluation.evaluators import SimpleClassificationEvaluator
from pioneerml.common.data_loader import BatchBundle
from pioneerml.common.pipeline.steps import BaseEvaluationStep


class GroupClassifierEvaluateStep(BaseEvaluationStep):
    step_key = "evaluate"

    def default_config(self) -> dict:
        return {
            "threshold": 0.5,
            "metrics": ["binary_classification_from_tensors"],
            "plots": ["loss_curves"],
            "batch_size": 1,
            "chunk_row_groups": 4,
            "chunk_workers": None,
        }

    def build_evaluator(self):
        return SimpleClassificationEvaluator()

    def evaluate_from_loader(self, *, loader, cfg: dict[str, Any], module, evaluator) -> dict[str, Any]:
        module.eval()
        device = next(module.parameters()).device
        total_loss = 0.0
        total_samples = 0
        preds_all: list[torch.Tensor] = []
        targets_all: list[torch.Tensor] = []
        threshold = float(cfg.get("threshold", 0.5))

        with torch.no_grad():
            for batch in loader:
                batch = batch.to(device, non_blocking=True)
                logits = module(batch)
                loss, _ = module.compute_loss(logits, batch)
                preds = module.primary_predictions(logits)
                target = module.primary_target(batch, preds)
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
        loss = float(total_loss / total_samples)

        plot_path = evaluator.resolve_plot_path(cfg)
        plot_outputs = self.apply_registered_plots(
            context={
                "plot_kwargs_by_name": {
                    "loss_curves": {
                        "train_losses": module,
                        "save_path": plot_path,
                        "show": False,
                    }
                }
            },
            plot_names=self.resolve_plot_names(cfg),
        )
        train_history, train_total = evaluator.concise_history(list(module.train_epoch_loss_history))
        val_history, val_total = evaluator.concise_history(list(module.val_epoch_loss_history))
        metrics = {
            "loss": loss,
            "threshold": float(threshold),
            "train_loss_history": train_history,
            "train_loss_history_total_points": train_total,
            "val_loss_history": val_history,
            "val_loss_history_total_points": val_total,
            "loss_plot_path": plot_outputs.get("loss_curves_path"),
        }
        self.apply_registered_metrics(
            metrics=metrics,
            context={
                "preds_binary": preds_binary,
                "targets": target_float,
            },
            metric_names=self.resolve_metric_names(cfg),
        )
        return metrics

@step(name="evaluate_group_classifier")
def evaluate_group_classifier_step(
    module: Any,
    dataset: BatchBundle,
    pipeline_config: dict | None = None,
) -> dict:
    return GroupClassifierEvaluateStep(pipeline_config=pipeline_config).execute(
        payloads={"module": module, "loader": dataset}
    )
