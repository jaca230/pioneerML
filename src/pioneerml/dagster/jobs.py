"""
Dagster jobs for training and evaluating PIONEER ML models.

These jobs demonstrate how to orchestrate training and inference with Dagster using
the existing GraphLightningModule and GraphDataModule utilities.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict

import torch
from dagster import In, Out, job, op
from torch_geometric.data import Data

from pioneerml.evaluation import multilabel_classification_metrics
from pioneerml.models import GroupClassifier
from pioneerml.training import GraphDataModule, GraphLightningModule


@dataclass
class SyntheticDataConfig:
    num_samples: int = 256
    num_nodes: int = 16
    num_classes: int = 3
    batch_size: int = 16
    val_split: float = 0.25


@op(out=Out(GraphDataModule), config_schema={"num_samples": int, "num_nodes": int, "num_classes": int, "batch_size": int, "val_split": float})
def build_synthetic_datamodule(context) -> GraphDataModule:
    cfg = SyntheticDataConfig(**context.op_config)

    def make_record(num_nodes: int, num_classes: int) -> Data:
        class_offsets = torch.tensor(
            [
                [1.0, 0.0, 0.0, 0.0, 0.0],  # pi-ish
                [0.0, 1.0, 0.0, 0.0, 0.0],  # mu-ish
                [0.0, 0.0, 1.0, 0.0, 0.0],  # e+ ish
            ]
        )
        label = torch.randint(0, num_classes, (1,)).item()
        x = torch.randn(num_nodes, 5) * 1.2 + class_offsets[label]
        edge_index = torch.randint(0, num_nodes, (2, num_nodes * 2))
        edge_attr = torch.randn(edge_index.shape[1], 4)
        y = torch.zeros(num_classes)
        y[label] = 1.0
        return Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y)

    records = [make_record(cfg.num_nodes, cfg.num_classes) for _ in range(cfg.num_samples)]
    return GraphDataModule(dataset=records, val_split=cfg.val_split, batch_size=cfg.batch_size)


@op(out=Out(GraphLightningModule), config_schema={"num_classes": int, "lr": float})
def build_model_module(context) -> GraphLightningModule:
    num_classes = context.op_config.get("num_classes", 3)
    lr = context.op_config.get("lr", 5e-4)
    model = GroupClassifier(num_classes=num_classes)
    return GraphLightningModule(model, task="classification", lr=lr)


@op(
    ins={
        "module": In(GraphLightningModule),
        "datamodule": In(GraphDataModule),
    },
    out=Out(dict),
    config_schema={
        "trainer_params": dict,
        "checkpoint_path": str,
    },
)
def train_lightning(context, module: GraphLightningModule, datamodule: GraphDataModule) -> Dict:
    import pytorch_lightning as pl

    trainer_params = context.op_config.get("trainer_params", {})
    trainer_params.setdefault("accelerator", "cpu")
    trainer_params.setdefault("max_epochs", 5)
    trainer_params.setdefault("limit_train_batches", 5)
    trainer_params.setdefault("limit_val_batches", 1)
    checkpoint_path = Path(context.op_config.get("checkpoint_path", "outputs/dagster/train_model.ckpt"))
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

    trainer = pl.Trainer(**trainer_params)
    trainer.fit(module, datamodule=datamodule)

    metrics = {k: float(v) if hasattr(v, "item") else v for k, v in trainer.logged_metrics.items()}
    trainer.save_checkpoint(checkpoint_path)

    return {
        "checkpoint_path": str(checkpoint_path),
        "metrics": metrics,
    }


@op(
    ins={"checkpoint_info": In(dict), "datamodule": In(GraphDataModule)},
    out=Out(dict),
    config_schema={"num_classes": int},
)
def run_inference(context, checkpoint_info: Dict, datamodule: GraphDataModule) -> Dict:
    checkpoint_path = checkpoint_info.get("checkpoint_path")
    if not checkpoint_path:
        raise ValueError("checkpoint_path missing from training output")

    module = GraphLightningModule.load_from_checkpoint(
        checkpoint_path,
        model=GroupClassifier(num_classes=context.op_config.get("num_classes", 3)),
        task="classification",
    ).eval()

    preds_list, targets_list = [], []
    loader = datamodule.val_dataloader()
    if isinstance(loader, list):
        loader = loader[0]
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(module.device) if hasattr(module, "device") else batch
            out = module(batch)
            preds_list.append(out.cpu())
            targets_list.append(batch.y.cpu())

    preds = torch.cat(preds_list)
    targets = torch.cat(targets_list)
    metrics = multilabel_classification_metrics(preds, targets, class_names=["pi", "mu", "e+"])
    return {"checkpoint_path": checkpoint_path, "inference_metrics": metrics}


@job
def dagster_train_job():
    dm = build_synthetic_datamodule()
    _ = train_lightning(build_model_module(), dm)


@job
def dagster_train_and_infer_job():
    dm = build_synthetic_datamodule()
    ckpt = train_lightning(build_model_module(), dm)
    _ = run_inference(ckpt, dm)
