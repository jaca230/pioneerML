from torch.utils.data import DataLoader
from zenml import step

from pioneerml.common.pipeline_utils.evaluation import SimpleRegressionEvaluator
from pioneerml.pipelines.training.endpoint_regression.dataset import EndpointRegressorDataset
from pioneerml.pipelines.training.endpoint_regression.steps.config import resolve_step_config
from pioneerml.pipelines.training.endpoint_regression.steps.train import _collate_graphs, _split_dataset_to_graphs


_EVALUATOR = SimpleRegressionEvaluator()


@step
def evaluate_endpoint_regressor(
    module,
    dataset: EndpointRegressorDataset,
    pipeline_config: dict | None = None,
) -> dict:
    step_config = resolve_step_config(pipeline_config, "evaluate")
    graphs = _split_dataset_to_graphs(dataset)
    batch_size = int(step_config.get("batch_size", 1)) if step_config else 1
    return _EVALUATOR.evaluate(
        module=module,
        graphs=graphs,
        batch_size=batch_size,
        loader_cls=DataLoader,
        collate_fn=_collate_graphs,
        plot_config=step_config,
    )
