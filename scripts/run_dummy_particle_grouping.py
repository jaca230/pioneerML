"""
Standalone runner for the dummy particle grouping ZenML pipeline.

Use this to validate the pipeline outside of notebooks before copying the
flow into an example notebook.
"""

from pioneerml.zenml import load_step_output
from pioneerml.zenml import utils as zenml_utils
from pioneerml.zenml.pipelines import dummy_particle_grouping_pipeline


def main() -> None:
    root = zenml_utils.find_project_root()
    zenml_client = zenml_utils.setup_zenml_for_notebook(root_path=root, use_in_memory=True)
    print(f"ZenML ready with stack: {zenml_client.active_stack_model.name}")

    run = dummy_particle_grouping_pipeline.with_options(enable_cache=False)()
    print(f"Run name: {run.name}")
    print(f"Run status: {run.status}")

    module = load_step_output(run, "train_dummy_module")
    datamodule = load_step_output(run, "build_dummy_datamodule")
    preds = load_step_output(run, "collect_dummy_predictions", index=0)
    targets = load_step_output(run, "collect_dummy_predictions", index=1)

    if module is not None:
        device = next(module.parameters()).device
        print(f"Trained module device: {device}")

    if datamodule is not None:
        datamodule.setup(stage="fit")
        val_len = len(datamodule.val_dataset) if datamodule.val_dataset is not None else 0
        train_len = len(datamodule.train_dataset) if datamodule.train_dataset is not None else 0
        print(f"Train samples: {train_len}, Val samples: {val_len}")

    if preds is not None and targets is not None:
        print(f"Preds shape: {tuple(preds.shape)}, Targets shape: {tuple(targets.shape)}")


if __name__ == "__main__":
    main()
