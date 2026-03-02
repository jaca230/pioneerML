
from typing import Any

import torch
from zenml import step

from pioneerml.common.evaluation.evaluators import SimpleClassificationEvaluator
from pioneerml.common.loader import GroupClassifierGraphLoaderFactory, TrainingBatchBundle
from pioneerml.common.pipeline.steps import BaseEvaluationStep, BaseLoaderStep


class GroupClassifierEvaluateStep(BaseEvaluationStep):
    step_key = "evaluate"

    def __init__(self, *, module: Any, dataset: TrainingBatchBundle, pipeline_config: dict | None = None) -> None:
        super().__init__(pipeline_config=pipeline_config)
        self.module = module
        self.dataset = dataset
        self.loader_factory = BaseLoaderStep.ensure_loader_factory(dataset, expected_type=GroupClassifierGraphLoaderFactory)
        self.evaluator = SimpleClassificationEvaluator()

    def default_config(self) -> dict:
        return {
            "threshold": 0.5,
            "metrics": ["binary_classification_from_tensors"],
            "plots": ["loss_curves"],
            "batch_size": 1,
            "chunk_row_groups": 4,
            "chunk_workers": None,
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
        loss = float(total_loss / total_samples)

        plot_path = self.evaluator.resolve_plot_path(plot_config)
        plot_outputs = self.apply_registered_plots(
            context={
                "plot_kwargs_by_name": {
                    "loss_curves": {
                        "train_losses": self.module,
                        "save_path": plot_path,
                        "show": False,
                    }
                }
            },
            plot_names=self.resolve_plot_names(plot_config),
        )
        train_history, train_total = self.evaluator.concise_history(list(self.module.train_epoch_loss_history))
        val_history, val_total = self.evaluator.concise_history(list(self.module.val_epoch_loss_history))
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
            metric_names=self.resolve_metric_names(plot_config),
        )
        return metrics

    def execute(self) -> dict:
        cfg = self.get_config()
        params = BaseLoaderStep.resolve_loader_params(cfg, purpose="evaluate")
        raw_loader_cfg = cfg.get("loader_config")
        if isinstance(raw_loader_cfg, dict):
            if not isinstance(raw_loader_cfg.get("evaluate"), dict) and isinstance(raw_loader_cfg.get("val"), dict):
                params = BaseLoaderStep.resolve_loader_params(cfg, purpose="val")
        provider = self.loader_factory.build_loader(loader_params=params)
        if not provider.include_targets:
            raise RuntimeError("GroupClassifierGraphLoader must run in train mode for evaluation.")
        loader = provider.make_dataloader(shuffle_batches=False)
        metrics = self._evaluate_from_loader(loader=loader, threshold=float(cfg.get("threshold", 0.5)), plot_config=cfg)
        diag = BaseLoaderStep.log_loader_diagnostics(label="evaluate", loader_provider=provider)
        if diag:
            metrics["loader_diagnostics"] = diag
        return metrics


@step(name="evaluate_group_classifier")
def evaluate_group_classifier_step(
    module: Any,
    dataset: TrainingBatchBundle,
    pipeline_config: dict | None = None,
) -> dict:
    return GroupClassifierEvaluateStep(module=module, dataset=dataset, pipeline_config=pipeline_config).execute()
