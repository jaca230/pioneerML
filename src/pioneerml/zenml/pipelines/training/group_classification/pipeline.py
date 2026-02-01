import torch
import torch.nn as nn
import warnings
from zenml import pipeline, step

from pioneerml.models.classifiers import GroupClassifier
from pioneerml.training.lightning import GraphLightningModule
from pioneerml.zenml.utils import detect_available_accelerator
from pioneerml.zenml.materializers import GroupClassifierBatchMaterializer
from .loader import GroupClassifierBatch, load_group_classifier_batch


@step(enable_cache=False, output_materializers=GroupClassifierBatchMaterializer)
def load_group_classifier_data(
    parquet_paths: list[str],
    *,
    config_json: dict | None = None,
) -> GroupClassifierBatch:
    return load_group_classifier_batch(parquet_paths, config_json=config_json)


@step
def train_group_classifier(
    batch: GroupClassifierBatch,
    *,
    max_epochs: int = 10,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
) -> GraphLightningModule:
    warnings.filterwarnings(
        "ignore",
        message="The 'train_dataloader' does not have many workers.*",
        category=UserWarning,
    )
    warnings.filterwarnings(
        "ignore",
        message="isinstance\\(treespec, LeafSpec\\) is deprecated.*",
        category=DeprecationWarning,
    )
    warnings.filterwarnings(
        "ignore",
        message="isinstance\\(treespec, LeafSpec\\) is deprecated.*",
        category=UserWarning,
    )
    warnings.filterwarnings(
        "ignore",
        message="isinstance\\(treespec, LeafSpec\\) is deprecated.*",
        category=Warning,
        module="pytorch_lightning\\.utilities\\._pytree",
    )
    warnings.filterwarnings(
        "ignore",
        message="The 'val_dataloader' does not have many workers.*",
        category=UserWarning,
    )
    try:
        from zenml import get_step_context

        ctx = get_step_context()
        params = getattr(getattr(ctx, "pipeline_run", None), "config", None)
        if params is not None:
            overrides = params.parameters or {}
            if "max_epochs" in overrides:
                max_epochs = int(overrides["max_epochs"])
            if "lr" in overrides:
                lr = float(overrides["lr"])
            if "weight_decay" in overrides:
                weight_decay = float(overrides["weight_decay"])
    except Exception:
        pass
    model = GroupClassifier(num_classes=batch.targets.shape[-1])
    module = GraphLightningModule(
        model,
        task="classification",
        loss_fn=nn.BCEWithLogitsLoss(),
        lr=lr,
        weight_decay=weight_decay,
    )

    accelerator, devices = detect_available_accelerator()
    trainer = torch.utils.data.DataLoader(
        [batch.data],
        batch_size=1,
        shuffle=False,
        collate_fn=lambda items: items[0],
    )
    lightning_trainer = None
    # Lazy import to avoid Lightning dependency at import time.
    import pytorch_lightning as pl

    lightning_trainer = pl.Trainer(
        accelerator=accelerator,
        devices=devices,
        max_epochs=max_epochs,
        enable_checkpointing=False,
        logger=False,
    )
    lightning_trainer.fit(module, train_dataloaders=trainer, val_dataloaders=trainer)
    return module


@pipeline
def group_classification_pipeline(
    parquet_paths: list[str],
    *,
    config_json: dict | None = None,
    max_epochs: int = 10,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
):
    batch = load_group_classifier_data(parquet_paths, config_json=config_json)
    module = train_group_classifier(
        batch,
        max_epochs=max_epochs,
        lr=lr,
        weight_decay=weight_decay,
    )
    return module, batch
