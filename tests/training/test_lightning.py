"""
Tests for Lightning utilities and pipeline integration.
"""

import pytest
from torch_geometric.loader import DataLoader as GeoDataLoader

from pioneerml.models import GroupClassifier
from pioneerml.pipelines import Context, StageConfig
from pioneerml.pipelines.stages import LightningTrainStage
from pioneerml.training import GraphLightningModule, GraphDataModule
from pioneerml.data import GraphGroupDataset


def _make_batch(sample_graph_record):
    dataset = GraphGroupDataset([sample_graph_record], num_classes=3)
    loader = GeoDataLoader(dataset, batch_size=1)
    return next(iter(loader))


def test_graph_lightning_module_forward(sample_graph_record):
    batch = _make_batch(sample_graph_record)
    module = GraphLightningModule(GroupClassifier(num_classes=3), task="classification")

    output = module(batch)
    assert output.shape == (1, 3)

    loss = module.training_step(batch, 0)
    assert loss.dim() == 0


def test_graph_data_module_splits(sample_graph_record):
    dataset = GraphGroupDataset([sample_graph_record for _ in range(5)], num_classes=3)
    datamodule = GraphDataModule(dataset=dataset, val_split=0.2, test_split=0.0, batch_size=2)
    datamodule.setup()

    assert datamodule.train_dataset is not None
    assert len(datamodule.train_dataset) == 4
    assert datamodule.val_dataset is not None
    assert len(datamodule.val_dataset) == 1


@pytest.mark.filterwarnings("ignore::UserWarning")
def test_lightning_train_stage_runs(sample_graph_record):
    dataset = GraphGroupDataset([sample_graph_record for _ in range(4)], num_classes=3)
    datamodule = GraphDataModule(dataset=dataset, val_split=0.0, test_split=0.0, batch_size=2)
    module = GraphLightningModule(GroupClassifier(num_classes=3), task="classification")

    stage = LightningTrainStage(
        config=StageConfig(
            name="lightning_train",
            params={
                "datamodule": datamodule,
                "module": module,
                "trainer_params": {
                    "max_epochs": 1,
                    "limit_train_batches": 1,
                    "limit_val_batches": 0,
                    "logger": False,
                    "enable_checkpointing": False,
                },
            },
        )
    )

    context = Context()
    stage.execute(context)

    assert "lightning_module" in context
    assert "trainer" in context
