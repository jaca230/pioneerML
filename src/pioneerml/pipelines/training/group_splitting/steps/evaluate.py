from torch_geometric.loader import DataLoader
from zenml import step

from pioneerml.common.pipeline_utils.evaluation import SimpleClassificationEvaluator
from pioneerml.pipelines.training.group_splitting.dataset import GroupSplitterDataset
from pioneerml.pipelines.training.group_splitting.steps.config import resolve_step_config
from pioneerml.pipelines.training.group_splitting.steps.train import _split_dataset_to_graphs


_EVALUATOR = SimpleClassificationEvaluator()


@step
def evaluate_group_splitter(
    module,
    dataset: GroupSplitterDataset,
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
        plot_config=step_config,
    )
