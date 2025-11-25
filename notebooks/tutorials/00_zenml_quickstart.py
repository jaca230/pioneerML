# %% [markdown]
# # Tutorial 0: ZenML Quickstart
# 
# Run the smallest end-to-end ZenML pipeline in this repository, load the
# trained model and data module from the ZenML artifacts, and generate a few
# quick diagnostic plots.
# 
# What you'll see (with detailed interpretation guidance):
# - How to spin up ZenML in in-memory mode (no server, minimal local state).
# - A minimal training run on synthetic data using our `GroupClassifier`.
# - How to pull artifacts back out of ZenML and compute plots.
# - How to interpret each plot (axes, computation, and what “good” looks like).

# %%
from pathlib import Path

import torch

from pioneerml.evaluation.plots import (
    plot_multilabel_confusion_matrix,
    plot_precision_recall_curves,
    plot_roc_curves,
)
from pioneerml.training import plot_loss_curves
from pioneerml.zenml import load_step_output
from pioneerml.zenml import utils as zenml_utils
from pioneerml.zenml.pipelines import zenml_training_pipeline


def find_project_root() -> Path:
    """Walk upward to locate the repo root (pyproject or .git)."""
    start = Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd().resolve()
    for path in [start] + list(start.parents):
        if (path / "pyproject.toml").exists() or (path / ".git").exists():
            return path
    return start


PROJECT_ROOT = find_project_root()

zenml_client = zenml_utils.setup_zenml_for_notebook(root_path=PROJECT_ROOT, use_in_memory=True)
print(f"ZenML initialized with stack: {zenml_client.active_stack_model.name}")

# %% [markdown]
# ## Run the quickstart pipeline
# Execute the ZenML pipeline and load the trained Lightning module plus the
# synthetic data module from the run artifacts. ZenML stores step outputs as
# artifacts; `load_step_output` pulls them back so we can keep working in pure
# Python without re-running training.
# 
# Why this matters:
# - Running once, reloading many times keeps notebooks fast.
# - You can switch between CPU/GPU here; the stored artifact is device-agnostic
#   and will load onto whatever is available now.

# %%
run = zenml_training_pipeline.with_options(enable_cache=False)()
print(f"Pipeline run status: {run.status}")

trained_module = load_step_output(run, "train_module")
datamodule = load_step_output(run, "build_datamodule")

if trained_module is None or datamodule is None:
    raise RuntimeError("Could not load artifacts from the zenml_training_pipeline run.")

trained_module.eval()
datamodule.setup(stage="fit")
device = next(trained_module.parameters()).device
print(f"Loaded artifacts from run {run.name} (device={device})")

# %% [markdown]
# ## Collect predictions and targets
# Use the validation split (or training split if validation is missing) to
# gather predictions and targets for plotting.
# 
# We switch the module to eval mode (disables dropout/batch norm updates),
# iterate over the validation loader, and collect logits and labels on CPU so
# plotting stays lightweight. Predictions are *logits* (unnormalized scores);
# we’ll convert them inside each plotting function.

# %%
val_loader = datamodule.val_dataloader()
if isinstance(val_loader, list) and len(val_loader) == 0:
    val_loader = datamodule.train_dataloader()

preds, targets = [], []
for batch in val_loader:
    batch = batch.to(device)
    with torch.no_grad():
        preds.append(trained_module(batch).detach().cpu())
        targets.append(batch.y.detach().cpu())

predictions = torch.cat(preds)
targets = torch.cat(targets)
print(f"Collected predictions for {len(targets)} samples.")

# %% [markdown]
# ## Plot training diagnostics
# Four key diagnostics rendered inline (no files written):
# 
# 1) Loss curves (`plot_loss_curves`)
#    - X-axis: epoch index. Y-axis: loss value.
#    - Computation: stored `train_epoch_loss_history` / `val_epoch_loss_history`
#      tracked during Lightning training.
#    - Interpretation: steady downward trend on both curves → learning. If
#      train decreases while val increases, you're overfitting. Flat lines mean
#      the model is stuck.
# 
# 2) Confusion matrices (`plot_multilabel_confusion_matrix`)
#    - One small matrix per class: rows are true (negative/positive), columns
#      are predicted (negative/positive). Normalized so values sum to 1.
#    - Computation: logits → sigmoid → threshold at 0.5 → binary prediction per
#      class, compared against one-hot labels.
#    - Interpretation: dark diagonal = correct predictions. Off-diagonal mass
#      indicates false positives/negatives for that class.
# 
# 3) ROC curves (`plot_roc_curves`)
#    - X-axis: False Positive Rate. Y-axis: True Positive Rate.
#    - Computation: sweep thresholds over the sigmoid probabilities and measure
#      TPR/FPR for each class; area under curve (AUC) summarizes ranking quality.
#    - Interpretation: curves near the top-left and higher AUC are better. A
#      diagonal line means random guessing.
# 
# 4) Precision-Recall curves (`plot_precision_recall_curves`)
#    - X-axis: Recall. Y-axis: Precision.
#    - Computation: sweep thresholds over sigmoid probabilities and compute
#      precision/recall for each class; area under curve (Average Precision)
#      summarizes performance on imbalanced data.
#    - Interpretation: higher curves mean better precision while retrieving most
#      positives. If precision crashes at higher recall, the model struggles to
#      find positives without many false alarms.

# %%
plot_loss_curves(trained_module, title="Quickstart: Loss Curves", show=True)

plot_multilabel_confusion_matrix(
    predictions=predictions,
    targets=targets,
    class_names=["pi", "mu", "e+"],
    threshold=0.5,
    normalize=True,
    save_path=None,
    show=True,
)

plot_roc_curves(
    predictions=predictions,
    targets=targets,
    class_names=["pi", "mu", "e+"],
    save_path=None,
    show=True,
)

plot_precision_recall_curves(
    predictions=predictions,
    targets=targets,
    class_names=["pi", "mu", "e+"],
    save_path=None,
    show=True,
)
