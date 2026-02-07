from torch.utils.data import DataLoader
from zenml import step

from pioneerml.common.pipeline_utils.evaluation import SimpleClassificationEvaluator
from pioneerml.pipelines.training.group_classification_event.dataset import GroupClassifierEventDataset
from pioneerml.pipelines.training.group_classification_event.steps.config import resolve_step_config
from pioneerml.pipelines.training.group_classification_event.steps.train import (
    _collate_graphs,
    _split_dataset_to_graphs,
)


_EVALUATOR = SimpleClassificationEvaluator()


@step
def evaluate_group_classifier_event(
    module,
    dataset: GroupClassifierEventDataset,
    pipeline_config: dict | None = None,
) -> dict:
    step_config = resolve_step_config(pipeline_config, "evaluate")
    threshold = 0.5 if step_config is None else float(step_config.get("threshold", 0.5))
    graphs = _split_dataset_to_graphs(dataset)
    batch_size = int(step_config.get("batch_size", 1)) if step_config else 1
    return _EVALUATOR.evaluate(
        module=module,
        graphs=graphs,
        threshold=threshold,
        batch_size=batch_size,
        loader_cls=DataLoader,
        collate_fn=_collate_graphs,
        plot_config=step_config,
    )
