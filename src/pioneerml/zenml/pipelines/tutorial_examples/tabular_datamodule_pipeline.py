"""
ZenML pipeline demonstrating how to build a custom LightningDataModule
for non-graph tabular data and use it inside a ZenML pipeline.
"""

from dataclasses import dataclass
from typing import Optional

import pytorch_lightning as pl
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset, random_split
from zenml import pipeline, step

from pioneerml.zenml.materializers import TorchTensorMaterializer
from pioneerml.zenml.utils import detect_available_accelerator


@dataclass
class TabularConfig:
    num_samples: int = 400
    num_features: int = 8
    num_classes: int = 3
    batch_size: int = 32
    val_split: float = 0.2
    test_split: float = 0.1
    seed: int = 123


class TabularDataModule(pl.LightningDataModule):
    """Simple LightningDataModule for synthetic tabular classification."""

    def __init__(self, config: TabularConfig):
        super().__init__()
        self.config = config
        self.train_dataset: Optional[TensorDataset] = None
        self.val_dataset: Optional[TensorDataset] = None
        self.test_dataset: Optional[TensorDataset] = None

    def setup(self, stage: Optional[str] = None) -> None:
        if self.train_dataset is not None:
            return

        torch.manual_seed(self.config.seed)
        # Create clustered synthetic features per class
        class_offsets = torch.randn(self.config.num_classes, self.config.num_features) * 0.5
        labels = torch.randint(0, self.config.num_classes, (self.config.num_samples,))
        features = torch.randn(self.config.num_samples, self.config.num_features) * 0.6
        features += class_offsets[labels]
        targets = labels

        dataset = TensorDataset(features, targets)

        val_len = int(self.config.num_samples * self.config.val_split)
        test_len = int(self.config.num_samples * self.config.test_split)
        train_len = self.config.num_samples - val_len - test_len
        lengths = [train_len, val_len, test_len] if test_len > 0 else [train_len, val_len]

        generator = torch.Generator().manual_seed(self.config.seed)
        splits = random_split(dataset, lengths, generator=generator)

        self.train_dataset = splits[0]
        self.val_dataset = splits[1] if len(splits) > 1 and val_len > 0 else None
        self.test_dataset = splits[2] if len(splits) > 2 and test_len > 0 else None

    def _loader(self, dataset: Optional[TensorDataset], shuffle: bool) -> DataLoader:
        if dataset is None:
            return DataLoader([])
        return DataLoader(
            dataset,
            batch_size=self.config.batch_size,
            shuffle=shuffle,
            num_workers=0,
        )

    def train_dataloader(self) -> DataLoader:
        return self._loader(self.train_dataset, shuffle=True)

    def val_dataloader(self) -> DataLoader:
        return self._loader(self.val_dataset, shuffle=False)

    def test_dataloader(self) -> DataLoader:
        return self._loader(self.test_dataset, shuffle=False)


class TabularClassifier(pl.LightningModule):
    """Tiny MLP classifier for the synthetic tabular data."""

    def __init__(self, config: TabularConfig, lr: float = 1e-3):
        super().__init__()
        self.save_hyperparameters(ignore=["config"])
        self.config = config
        self.model = nn.Sequential(
            nn.Linear(config.num_features, 32),
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.ReLU(),
            nn.Linear(32, config.num_classes),
        )
        self.lr = lr

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = nn.functional.cross_entropy(logits, y)
        preds = logits.argmax(dim=1)
        acc = (preds == y).float().mean()
        self.log("train_loss", loss, prog_bar=True)
        self.log("train_acc", acc, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = nn.functional.cross_entropy(logits, y)
        preds = logits.argmax(dim=1)
        acc = (preds == y).float().mean()
        self.log("val_loss", loss, prog_bar=True)
        self.log("val_acc", acc, prog_bar=True)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)


@step
def build_tabular_datamodule(config: TabularConfig) -> TabularDataModule:
    dm = TabularDataModule(config)
    dm.setup(stage="fit")
    return dm


@step
def build_tabular_model(config: TabularConfig) -> TabularClassifier:
    return TabularClassifier(config=config, lr=1e-3)


@step(enable_cache=False)
def train_tabular_model(module: TabularClassifier, datamodule: TabularDataModule) -> TabularClassifier:
    accelerator, devices = detect_available_accelerator()
    trainer = pl.Trainer(
        accelerator=accelerator,
        devices=devices,
        max_epochs=10,
        limit_train_batches=5,
        limit_val_batches=2,
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=False,
    )
    trainer.fit(module, datamodule=datamodule)
    return module.eval()


@step(
    enable_cache=False,
    output_materializers=(TorchTensorMaterializer, TorchTensorMaterializer),
)
def evaluate_tabular_model(
    module: TabularClassifier, datamodule: TabularDataModule
) -> tuple[torch.Tensor, torch.Tensor]:
    datamodule.setup(stage="fit")
    loader = datamodule.val_dataloader() or datamodule.train_dataloader()
    preds, targets = [], []
    device = next(module.parameters()).device
    module.eval()
    for batch in loader:
        x, y = batch
        x = x.to(device)
        with torch.no_grad():
            logits = module(x).detach().cpu()
        preds.append(logits)
        targets.append(y.detach().cpu())
    return torch.cat(preds), torch.cat(targets)


@pipeline
def tabular_datamodule_pipeline(config=None):
    """Run the tabular DataModule pipeline.

    Args:
        config: Optional TabularConfig. If None, a default config is used.
    """
    if config is None:
        config = TabularConfig()
    datamodule = build_tabular_datamodule(config)
    model = build_tabular_model(config)
    trained = train_tabular_model(model, datamodule)
    preds, targets = evaluate_tabular_model(trained, datamodule)
    return trained, datamodule, preds, targets
