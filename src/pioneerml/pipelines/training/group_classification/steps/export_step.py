
from types import SimpleNamespace
from typing import Any

from zenml import step

from pioneerml.common.loader import GroupClassifierGraphLoaderFactory, BatchBundle
from pioneerml.common.pipeline.steps import BaseExportStep, BaseLoaderStep


class GroupClassifierExportStep(BaseExportStep):
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
        self.loader_factory = BaseLoaderStep.ensure_loader_factory(dataset, expected_type=GroupClassifierGraphLoaderFactory)
        self.hpo_params = hpo_params
        self.metrics = metrics

    def default_config(self) -> dict:
        return {}

    @staticmethod
    def _build_export_example(dataset: BatchBundle):
        data = dataset.data
        if hasattr(data, "node_graph_id"):
            return (data.x_node, data.edge_index, data.x_edge, data.node_graph_id)

        factory = getattr(dataset, "loader_factory", None) or getattr(dataset, "loader", None)
        if factory is not None:
            loader = factory.build_loader(
                loader_params={"mode": "train", "batch_size": 1, "chunk_row_groups": 1, "chunk_workers": 0}
            ).make_dataloader(shuffle_batches=False)
            for batch in loader:
                return (batch.x_node, batch.edge_index, batch.x_edge, batch.node_graph_id)
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
            default_export_dir="trained_models/groupclassifier",
            default_prefix="groupclassifier",
            example_builder=lambda _ignored: self._build_export_example(self.dataset),
        )


@step(name="export_group_classifier")
def export_group_classifier_step(
    module: Any,
    dataset: BatchBundle,
    pipeline_config: dict | None = None,
    hpo_params: dict | None = None,
    metrics: dict | None = None,
) -> dict:
    return GroupClassifierExportStep(
        module=module,
        dataset=dataset,
        pipeline_config=pipeline_config,
        hpo_params=hpo_params,
        metrics=metrics,
    ).execute()
