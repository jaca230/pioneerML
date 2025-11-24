"""
Evaluation and metrics pipeline stages.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Dict, Optional

from pioneerml.evaluation import MetricCollection, PLOT_REGISTRY, default_metrics_for_task
from pioneerml.pipelines.stage import StageConfig
from pioneerml.pipelines.stages.roles import EvaluatorStage


def _align_shapes(preds: Any, targets: Any) -> tuple[Any, Any]:
    """Align target shape to predictions when they have the same number of elements."""
    if preds is None or targets is None:
        return preds, targets

    def _numel(x: Any) -> int | None:
        if hasattr(x, "numel"):
            try:
                return int(x.numel())
            except Exception:
                pass
        if hasattr(x, "size"):
            size_attr = getattr(x, "size")
            if callable(size_attr):
                try:
                    return int(size_attr())
                except Exception:
                    pass
            else:
                try:
                    return int(size_attr)
                except Exception:
                    pass
        return None

    try:
        p_shape = preds.shape
        t_shape = targets.shape
        p_size = _numel(preds)
        t_size = _numel(targets)
        if p_size is not None and t_size is not None and p_size == t_size and p_shape != t_shape:
            targets = targets.reshape(p_shape)
    except Exception:
        pass
    return preds, targets


class EvaluateStage(EvaluatorStage):
    """
    Stage for evaluating model predictions with standardized metrics and plots.

    Params:
        task: "multilabel" (default) or "regression" to select default metrics.
        metrics: Optional list of metric names to compute (uses registry).
        metric_params: Extra kwargs passed to metric functions (e.g., threshold, class_names).
        plots: Optional list of plot names to generate (uses registry).
        plot_params: Extra kwargs for plotting functions.
        save_dir: Directory to write plot artifacts.
        metric_fn: Optional custom metric callable for backward compatibility.

    Example:
        >>> stage = EvaluateStage(
        ...     config=StageConfig(
        ...         name="evaluate",
        ...         inputs=["predictions", "targets"],
        ...         outputs=["evaluation_metrics"],
        ...         params={
        ...             "task": "multilabel",
        ...             "plots": ["multilabel_confusion", "precision_recall"],
        ...             "metric_params": {"threshold": 0.5, "class_names": ["pi", "mu", "e+"]},
        ...             "save_dir": "outputs/eval",
        ...         },
        ...     )
        ... )
    """

    def execute(self, context: Any) -> None:
        """Evaluate predictions with registry-backed metrics and plots."""
        params = self.config.params
        predictions = context.get(self.inputs[0] if self.inputs else "predictions")
        targets = context.get(self.inputs[1] if len(self.inputs) > 1 else "targets")

        if predictions is None:
            missing = self.inputs[0] if self.inputs else "predictions"
            raise KeyError(f"'{missing}' not found in context")

        predictions, targets = _align_shapes(predictions, targets)

        metric_fn: Optional[Callable] = params.get("metric_fn")
        task = params.get("task", "multilabel")
        metric_names = params.get("metrics")
        metric_params = params.get("metric_params", {}).copy()

        # Allow class_names to be passed once for all metrics
        class_names = params.get("class_names") or metric_params.get("class_names")
        if class_names and "class_names" not in metric_params:
            metric_params["class_names"] = class_names

        if metric_fn is not None:
            metrics = metric_fn(predictions, targets)
        else:
            names = metric_names or default_metrics_for_task(task)
            collection = MetricCollection.from_names(names)
            if targets is None:
                raise ValueError("Targets are required to compute evaluation metrics.")
            metrics = collection(predictions, targets, **metric_params)

        output_key = self.outputs[0] if self.outputs else "evaluation_metrics"
        context[output_key] = metrics

        # Optional plot generation
        plots = params.get("plots") or []
        if plots and targets is not None:
            save_dir = Path(params.get("save_dir", "outputs/evaluation"))
            save_dir.mkdir(parents=True, exist_ok=True)
            plot_params = params.get("plot_params", {}).copy()
            if class_names and "class_names" not in plot_params:
                plot_params["class_names"] = class_names

            generated: Dict[str, str | None] = {}
            for plot_name in plots:
                if plot_name not in PLOT_REGISTRY:
                    raise KeyError(f"Plot '{plot_name}' is not registered. Available: {list(PLOT_REGISTRY)}")
                plot_fn = PLOT_REGISTRY[plot_name]
                plot_path = save_dir / f"{plot_name}.png"
                generated[plot_name] = plot_fn(
                    predictions=predictions,
                    targets=targets,
                    save_path=plot_path,
                    **plot_params,
                )
            context["evaluation_plots"] = generated
