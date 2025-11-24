"""
Minimal quickstart to smoke-test training + evaluation without a notebook.

Creates synthetic graph data, trains a GroupClassifier for a couple of steps,
then runs EvaluateStage to produce metrics and plots under outputs/tutorial_quickstart.
"""

from __future__ import annotations

from pathlib import Path

import torch
from torch_geometric.data import Data

from pioneerml.models import GroupClassifier
from pioneerml.training import GraphDataModule, GraphLightningModule
from pioneerml.pipelines import Context, Pipeline, StageConfig
from pioneerml.pipelines.stages import EvaluateStage, LightningTrainStage


def make_synthetic_group(num_nodes: int = 16, num_classes: int = 3) -> Data:
    x = torch.randn(num_nodes, 5)
    edge_index = torch.randint(0, num_nodes, (2, num_nodes * 2))
    edge_attr = torch.randn(edge_index.shape[1], 4)
    y = torch.zeros(num_classes)
    y[torch.randint(0, num_classes, (1,))] = 1.0  # one-hot graph label
    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y)


def main() -> None:
    num_classes = 3
    records = [make_synthetic_group(num_classes=num_classes) for _ in range(64)]
    datamodule = GraphDataModule(dataset=records, val_split=0.25, batch_size=8)

    model = GroupClassifier(num_classes=num_classes)
    lightning_module = GraphLightningModule(model, task="classification", lr=5e-4)

    train_stage = LightningTrainStage(
        config=StageConfig(
            name="train",
            params={
                "module": lightning_module,
                "datamodule": datamodule,
                "trainer_params": {
                    "accelerator": "cpu",
                    "max_epochs": 1,
                    "limit_train_batches": 1,
                    "limit_val_batches": 1,
                    "logger": False,
                    "enable_checkpointing": False,
                },
            },
        )
    )

    pipeline = Pipeline([train_stage], name="quickstart")
    ctx = pipeline.run(Context())

    # Collect val predictions/targets
    val_loader = datamodule.val_dataloader()
    preds, targets = [], []
    module = ctx["lightning_module"].eval()
    with torch.no_grad():
        for batch in val_loader:
            batch = batch.to(module.device) if hasattr(module, "device") else batch
            out = module(batch)
            preds.append(out.cpu())
            targets.append(batch.y.cpu())
    preds = torch.cat(preds)
    targets = torch.cat(targets)

    eval_stage = EvaluateStage(
        config=StageConfig(
            name="eval",
            inputs=["preds", "targets"],
            outputs=["metrics"],
            params={
                "task": "multilabel",
                "plots": ["multilabel_confusion", "precision_recall"],
                "save_dir": "outputs/tutorial_quickstart",
                "metric_params": {"class_names": ["pi", "mu", "e+"]},
            },
        )
    )

    eval_ctx = Context({"preds": preds, "targets": targets})
    eval_stage.execute(eval_ctx)

    Path("outputs/tutorial_quickstart").mkdir(parents=True, exist_ok=True)
    print("Metrics:", eval_ctx["metrics"])
    print("Artifacts written to outputs/tutorial_quickstart")


if __name__ == "__main__":
    main()
