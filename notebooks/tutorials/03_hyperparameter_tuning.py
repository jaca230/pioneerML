# %% [markdown]
# # Tutorial 3: Hyperparameter Tuning with Optuna
# 
# Run a ZenML pipeline that performs a short Optuna sweep and then trains a
# model with the best parameters found.

# %%
from pathlib import Path

import torch

from pioneerml.zenml import load_step_output
from pioneerml.zenml import utils as zenml_utils
from pioneerml.zenml.pipelines import hyperparameter_tuning_pipeline

if "__file__" in globals():
    PROJECT_ROOT = Path(__file__).resolve().parents[2]
else:
    PROJECT_ROOT = Path.cwd().resolve()

zenml_client = zenml_utils.setup_zenml_for_notebook(root_path=PROJECT_ROOT, use_in_memory=True)
print(f"ZenML ready. Stack: {zenml_client.active_stack_model.name}")

# %% [markdown]
# ## Run the tuning pipeline
# The pipeline generates synthetic data, runs a small Optuna search, and trains
# a final model with the best hyperparameters.

# %%
run = hyperparameter_tuning_pipeline.with_options(enable_cache=False)(n_trials=2)
print(f"Pipeline run {run.name} status: {run.status}")

trained_module = load_step_output(run, "train_with_best_params")
datamodule = load_step_output(run, "prepare_tuning_datamodule")
best_params = load_step_output(run, "hyperparameter_search")

if trained_module is None or datamodule is None or best_params is None:
    raise RuntimeError("Could not load artifacts from the hyperparameter_tuning_pipeline run.")

trained_module.eval()
datamodule.setup(stage="fit")

print("Best parameters found:")
for key, value in best_params.items():
    print(f"- {key}: {value}")

# %% [markdown]
# ## Evaluate the tuned model
# Run a quick accuracy check on the validation split using the tuned model.

# %%
device = next(trained_module.parameters()).device
val_loader = datamodule.val_dataloader()
if isinstance(val_loader, list) and len(val_loader) == 0:
    val_loader = datamodule.train_dataloader()

correct = 0
total = 0
for batch in val_loader:
    batch = batch.to(device)
    with torch.no_grad():
        logits = trained_module(batch)
    preds = torch.argmax(logits, dim=1)
    labels = torch.argmax(batch.y, dim=1)
    correct += int((preds == labels).sum().item())
    total += int(labels.numel())

accuracy = correct / total if total else 0.0
print(f"Tuned model validation accuracy: {accuracy:.3f}")
