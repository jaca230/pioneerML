# %% [markdown]
# # Tutorial 1: Building ZenML Pipelines
# 
# Learn how our tutorial pipelines are structured, trigger a run, and inspect
# the outputs produced by ZenML.
# 
# What you'll see:
# - How to define ZenML steps/pipelines inline (no hidden magic).
# - How materializers control artifact serialization to avoid noisy warnings.
# - What each step contributes and how ZenML wires inputs/outputs.
# - A quick sanity-check metric and how to interpret the basic training run.
# 
# Reading the notebook:
# - Markdown above each block explains inputs/outputs and why the code is there.
# - Pay attention to device placement when reloading artifacts (CPU vs GPU).
# - Treat the final accuracy as a smoke test, not a benchmark (synthetic data).

# %%
from pathlib import Path

import torch

from torch_geometric.data import Data
from zenml import pipeline, step
from zenml.utils import source_utils

from pioneerml.models import GroupClassifier
from pioneerml.training import GraphDataModule, GraphLightningModule
from pioneerml.zenml.materializers import (
    GraphDataModuleMaterializer,
    PyGDataListMaterializer,
)
from pioneerml.zenml.utils import detect_available_accelerator, load_step_output
import pioneerml.zenml.utils as zenml_utils

def find_project_root() -> Path:
    """Walk upward to locate the repo root (pyproject or .git)."""
    start = Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd().resolve()
    for path in [start] + list(start.parents):
        if (path / "pyproject.toml").exists() or (path / ".git").exists():
            return path
    return start


PROJECT_ROOT = find_project_root()
source_utils.set_custom_source_root(PROJECT_ROOT / "src")

zenml_client = zenml_utils.setup_zenml_for_notebook(root_path=PROJECT_ROOT, use_in_memory=True)
print(f"Active ZenML stack: {zenml_client.active_stack_model.name}")

# %% [markdown]
# ## Define the basic training pipeline
# Build the steps directly in the notebook so you can see how ZenML pipelines
# are composed. The cell below walks through every piece:
# - `create_data`: synthetic graphs for a 3-class task, saved with a custom
#   materializer to avoid pickle spam.
# - `create_datamodule`: splits into train/val and sets sane batch/worker sizes.
# - `create_model` / `create_lightning_module`: model + Lightning wrapper.
# - `train_model`: short CPU/GPU-friendly fit with caching disabled.
# 
# Inputs/outputs:
# - Steps communicate via return values; ZenML handles wiring.
# - Materializers specify how to serialize outputs; here we use lightweight
#   torch saves to keep artifact warnings quiet.

# %%

def create_simple_synthetic_data(num_samples: int = 200) -> list[Data]:
    data = []
    for _ in range(num_samples):
        num_nodes = torch.randint(4, 8, (1,)).item()
        x = torch.randn(num_nodes, 5)
        edge_index = torch.randint(0, num_nodes, (2, num_nodes * 2))
        edge_attr = torch.randn(edge_index.shape[1], 4)
        label = torch.randint(0, 3, (1,)).item()
        y = torch.zeros(3)
        y[label] = 1.0
        data.append(Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y))
    return data


@step(output_materializers=PyGDataListMaterializer, enable_cache=False)
def create_data() -> list[Data]:
    return create_simple_synthetic_data()


@step(output_materializers=GraphDataModuleMaterializer, enable_cache=False)
def create_datamodule(data: list[Data]) -> GraphDataModule:
    return GraphDataModule(dataset=data, val_split=0.2, batch_size=32, num_workers=2)


@step
def create_model(num_classes: int = 3) -> GroupClassifier:
    return GroupClassifier(num_classes=num_classes, hidden=64, num_blocks=1)


@step
def create_lightning_module(model: GroupClassifier) -> GraphLightningModule:
    return GraphLightningModule(model, task="classification", lr=1e-3)


@step
def train_model(lightning_module: GraphLightningModule, datamodule: GraphDataModule) -> GraphLightningModule:
    import pytorch_lightning as pl

    accelerator, devices = detect_available_accelerator()

    trainer = pl.Trainer(
        accelerator=accelerator,
        devices=devices,
        max_epochs=3,
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=False,
    )
    trainer.fit(lightning_module, datamodule=datamodule)
    return lightning_module.eval()


@pipeline
def basic_training_pipeline_demo():
    data = create_data()
    datamodule = create_datamodule(data)
    model = create_model()
    lightning_module = create_lightning_module(model)
    trained_module = train_model(lightning_module, datamodule)
    return trained_module, datamodule


# Reuse the packaged pipeline implementation for execution while keeping the
# definition above for learning purposes.
from pioneerml.zenml.pipelines.tutorial_examples.basic_training import (
    basic_training_pipeline as packaged_basic_training_pipeline,
)

# %%
run = packaged_basic_training_pipeline.with_options(enable_cache=False)()
print(f"Pipeline run {run.name} status: {run.status}")

# Load artifacts from the run so we can inspect them locally
trained_module = load_step_output(run, "train_model")
datamodule = load_step_output(run, "create_datamodule")
model = getattr(trained_module, "model", None) if trained_module is not None else None

if trained_module is None or datamodule is None:
    raise RuntimeError("Could not load artifacts from the basic_training_pipeline run.")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
trained_module = trained_module.to(device).eval()
datamodule.setup(stage="fit")

# %% [markdown]
# ## Inspect the outputs
# Check the dataset size, batch shape, and key model parameters. This helps
# validate that the data/materializers round-tripped correctly and that the
# model config is what we expect.
# - Dataset size: confirms split sizes match expectations.
# - Batch shape: confirms node/edge dimensions align with model input.
# - Model config: sanity check on hidden dim and class count.

# %%
device = next(trained_module.parameters()).device
train_loader = datamodule.train_dataloader()
first_batch = next(iter(train_loader))
train_size = len(datamodule.train_dataset) if datamodule.train_dataset is not None else 0
val_size = len(datamodule.val_dataset) if datamodule.val_dataset is not None else 0

print("Training summary:")
print(f"- Run: {run.name}")
print(f"- Device: {device}")
print(f"- Dataset size: {train_size + val_size} samples (train={train_size}, val={val_size})")
print(f"- Batch shape: x={tuple(first_batch.x.shape)}, edge_index={tuple(first_batch.edge_index.shape)}")

if model:
    print("Model configuration:")
    print(f"- Type: {type(model).__name__}")
    print(f"- Hidden dimension: {getattr(model, 'hidden', 'n/a')}")
    print(f"- Num classes: {getattr(model, 'num_classes', 'n/a')}")

# %% [markdown]
# ## Evaluate quickly on the validation split
# Use the trained module to compute a lightweight accuracy metric so we have
# a sanity check that training worked. Accuracy here is on a tiny synthetic
# validation set, so treat it as a smoke test rather than a real benchmark.
# 
# Metric details:
# - Inputs: logits from the model vs one-hot labels.
# - Computation: argmax over logits and labels → class IDs → compare for exact
#   match, averaged over the validation set.
# - Interpretation: >0.33 means the model is learning above random (3 classes).

# %%
val_loader = datamodule.val_dataloader()
if isinstance(val_loader, list) and len(val_loader) == 0:
    val_loader = datamodule.train_dataloader()

correct = 0
total = 0
for batch in val_loader:
    batch = batch.to(device)
    with torch.no_grad():
        logits = trained_module(batch)
    labels = batch.y
    if labels.dim() == 1 and logits.shape[-1] > 0 and labels.numel() % logits.shape[-1] == 0:
        labels = labels.view(-1, logits.shape[-1])
    if labels.dim() > 1:
        labels = torch.argmax(labels, dim=1)
    preds = torch.argmax(logits, dim=1)
    correct += int((preds == labels).sum().item())
    total += int(labels.numel())

accuracy = correct / total if total else 0.0
print(f"Validation accuracy (quick check): {accuracy:.3f}")
