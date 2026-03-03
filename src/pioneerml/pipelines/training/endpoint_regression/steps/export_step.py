from types import SimpleNamespace
from typing import Any

from zenml import step

from pioneerml.common.loader import EndpointRegressionGraphLoaderFactory, BatchBundle
from pioneerml.common.pipeline.steps import BaseExportStep, BaseLoaderStep


class EndpointRegressorExportStep(BaseExportStep):
    step_key = "export"

    def __init__(
        self,
        *,
        module: Any,
        dataset: BatchBundle,
        pipeline_config: dict | None = None,
        hpo_params: dict | None = None,
        metrics: dict | None = None,
    ) -> None:
        super().__init__(pipeline_config=pipeline_config)
        self.module = module
        self.dataset = dataset
        self.loader_factory = BaseLoaderStep.ensure_loader_factory(
            dataset,
            expected_type=EndpointRegressionGraphLoaderFactory,
        )
        self.hpo_params = hpo_params
        self.metrics = metrics

    def default_config(self) -> dict:
        return {}

    @staticmethod
    def _to_export_example(batch):
        x = batch.x_node
        x_graph = batch.x_graph
        group_probs = x_graph[:, :3]
        u = x_graph[:, 3:4] if x_graph.shape[1] >= 4 else x.new_zeros((x.shape[0], 1))
        splitter_probs = x.new_zeros((x.shape[0], 0))
        return (
            x,
            batch.edge_index,
            batch.x_edge,
            batch.node_graph_id,
            u,
            group_probs,
            splitter_probs,
        )

    def _build_export_example(self):
        data = self.dataset.data
        if hasattr(data, "node_graph_id"):
            return self._to_export_example(data)

        factory = getattr(self.dataset, "loader_factory", None) or getattr(self.dataset, "loader", None)
        if factory is not None:
            loader = factory.build_loader(
                loader_params={
                    "mode": "train",
                    "batch_size": 1,
                    "chunk_row_groups": 1,
                    "chunk_workers": 0,
                }
            ).make_dataloader(shuffle_batches=False)
            for batch in loader:
                return self._to_export_example(batch)
        return None

    def execute(self) -> dict:
        cfg = self.get_config()
        params = BaseLoaderStep.resolve_loader_params(
            {
                "batch_size": 1,
                "chunk_row_groups": 1,
                "chunk_workers": 0,
                "loader_config": cfg.get("loader_config"),
            },
            purpose="export",
            forced_batch_size=1,
        )
        loader = self.loader_factory.build_loader(loader_params=params)
        data = loader.empty_data()
        data.source_parquet_paths = list(loader.parquet_paths)

        dataset_for_export = SimpleNamespace(data=data)
        return self.export_torchscript(
            module=self.module,
            dataset=dataset_for_export,
            cfg=cfg,
            pipeline_config=self.pipeline_config,
            hpo_params=self.hpo_params,
            metrics=self.metrics,
            default_export_dir="trained_models/endpoint_regressor",
            default_prefix="endpoint_regressor",
            example_builder=lambda _ignored: self._build_export_example(),
        )


@step(name="export_endpoint_regressor")
def export_endpoint_regressor_step(
    module: Any,
    dataset: BatchBundle,
    pipeline_config: dict | None = None,
    hpo_params: dict | None = None,
    metrics: dict | None = None,
) -> dict:
    return EndpointRegressorExportStep(
        module=module,
        dataset=dataset,
        pipeline_config=pipeline_config,
        hpo_params=hpo_params,
        metrics=metrics,
    ).execute()
