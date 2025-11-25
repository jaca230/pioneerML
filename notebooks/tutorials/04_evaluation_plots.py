# %% [markdown]
# # Tutorial 4: Evaluation and Custom Plots
# 
# Train a small model, gather predictions, compute metrics, and save standard
# evaluation plots using the built-in plotting utilities.

# %%
from pathlib import Path

import torch

from pioneerml.zenml import load_step_output
from pioneerml.zenml import utils as zenml_utils
from pioneerml.zenml.pipelines import evaluation_examples_pipeline

if "__file__" in globals():
    PROJECT_ROOT = Path(__file__).resolve().parents[2]
else:
    PROJECT_ROOT = Path.cwd().resolve()

zenml_client = zenml_utils.setup_zenml_for_notebook(root_path=PROJECT_ROOT, use_in_memory=True)
print(f"ZenML stack: {zenml_client.active_stack_model.name}")

# %% [markdown]
# ## Run the evaluation pipeline
# The pipeline trains a model, collects predictions on the validation split, and
# generates a few standard plots (confusion matrices, ROC, PR).

# %%
run = evaluation_examples_pipeline.with_options(enable_cache=False)()
print(f"Pipeline run {run.name} status: {run.status}")

trained_module = load_step_output(run, "train_evaluation_model")
datamodule = load_step_output(run, "prepare_evaluation_datamodule")
predictions_and_targets = load_step_output(run, "collect_predictions")
metrics = load_step_output(run, "compute_custom_metrics")
plot_paths = load_step_output(run, "generate_evaluation_plots")

if trained_module is None or datamodule is None:
    raise RuntimeError("Could not load required artifacts from the evaluation_examples_pipeline run.")

trained_module.eval()
datamodule.setup(stage="fit")

# %% [markdown]
# ## Pull predictions and metrics
# Load the predictions/targets from the ZenML artifact (fallback to recomputing
# them if needed), then print the summary metrics returned by the pipeline.

# %%
device = next(trained_module.parameters()).device

if predictions_and_targets is not None:
    predictions, targets = predictions_and_targets
else:
    val_loader = datamodule.val_dataloader()
    if isinstance(val_loader, list) and len(val_loader) == 0:
        val_loader = datamodule.train_dataloader()

    preds, trgs = [], []
    for batch in val_loader:
        batch = batch.to(device)
        with torch.no_grad():
            preds.append(trained_module(batch).detach().cpu())
            trgs.append(batch.y.detach().cpu())
    predictions = torch.cat(preds)
    targets = torch.cat(trgs)

print("Metrics:")
if metrics:
    for key, value in metrics.items():
        print(f"- {key}: {value}")
else:
    print("- No metrics artifact found.")

# %% [markdown]
# ## Plot locations
# The evaluation step already saved plots to disk. Surface the paths here so
# they are easy to find from the notebook or CLI.

# %%
if plot_paths:
    print("Plot paths:")
    for name, path in plot_paths.items():
        print(f"- {name}: {path}")
else:
    print("No plot paths were recorded.")
