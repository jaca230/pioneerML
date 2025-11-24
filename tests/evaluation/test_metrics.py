import numpy as np
import torch

from pioneerml.evaluation.metrics import (
    MetricCollection,
    default_metrics_for_task,
    multilabel_classification_metrics,
)
from pioneerml.pipelines import Context, StageConfig
from pioneerml.pipelines.stages import EvaluateStage


def test_multilabel_metric_shapes():
    preds = np.array([[2.0, -1.0], [-2.0, 3.0]])
    targets = np.array([[1, 0], [0, 1]])

    metrics = multilabel_classification_metrics(preds, targets, threshold=0.5, class_names=["pi", "mu"])

    assert metrics["subset_accuracy"] == 1.0
    assert metrics["precision"] >= 0.9
    assert metrics["class/pi/recall"] == 1.0
    assert metrics["class/mu/precision"] == 1.0


def test_evaluate_stage_computes_metrics_and_plots(tmp_path):
    preds = torch.tensor([[2.0, -1.0], [-2.0, 3.0]])
    targets = torch.tensor([[1.0, 0.0], [0.0, 1.0]])

    stage = EvaluateStage(
        config=StageConfig(
            name="eval",
            inputs=["preds", "targets"],
            outputs=["metrics"],
            params={
                "plots": ["multilabel_confusion"],
                "save_dir": tmp_path,
                "metric_params": {"class_names": ["pi", "mu"]},
            },
        )
    )

    context = Context({"preds": preds, "targets": targets})
    stage.execute(context)

    metrics = context["metrics"]
    assert metrics["multilabel_classification.subset_accuracy"] == 1.0

    plot_path = tmp_path / "multilabel_confusion.png"
    assert plot_path.exists()
