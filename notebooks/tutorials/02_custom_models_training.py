# %% [markdown]
# # Tutorial 2: Custom Models and Training
# 
# Train a custom Graph Convolutional Network (GCN) inside a ZenML pipeline and
# inspect the resulting model and data pipeline.

# %%
from pathlib import Path

import torch

from pioneerml.zenml import load_step_output
from pioneerml.zenml import utils as zenml_utils
from pioneerml.zenml.pipelines import custom_model_pipeline

if "__file__" in globals():
    PROJECT_ROOT = Path(__file__).resolve().parents[2]
else:
    PROJECT_ROOT = Path.cwd().resolve()

zenml_client = zenml_utils.setup_zenml_for_notebook(root_path=PROJECT_ROOT, use_in_memory=True)
print(f"ZenML configured. Active stack: {zenml_client.active_stack_model.name}")

# %% [markdown]
# ## Run the custom model pipeline
# This pipeline builds synthetic graphs, wraps them in a DataModule, defines a
# small `SimpleGCN`, and trains it for a few epochs.

# %%
run = custom_model_pipeline.with_options(enable_cache=False)()
print(f"Pipeline run {run.name} status: {run.status}")

trained_module = load_step_output(run, "train_custom_model")
datamodule = load_step_output(run, "create_custom_datamodule")
custom_model = load_step_output(run, "create_custom_model")

if trained_module is None or datamodule is None or custom_model is None:
    raise RuntimeError("Could not load artifacts from the custom_model_pipeline run.")

trained_module.eval()
datamodule.setup(stage="fit")

# %% [markdown]
# ## Explore the custom model
# Review the parameter count and the graph shapes flowing through the network.

# %%
device = next(trained_module.parameters()).device
param_count = sum(p.numel() for p in trained_module.parameters())

train_loader = datamodule.train_dataloader()
first_batch = next(iter(train_loader))

print("Custom model summary:")
print(f"- Run: {run.name}")
print(f"- Device: {device}")
print(f"- Parameters: {param_count:,}")
print(f"- Input nodes: {first_batch.x.shape[0]}, features per node: {first_batch.x.shape[1]}")

# %% [markdown]
# ## Quick validation accuracy
# Compute a simple accuracy metric to confirm the custom model trains end-to-end.

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
print(f"Validation accuracy (GCN): {accuracy:.3f}")
