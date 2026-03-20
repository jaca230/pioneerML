
from types import SimpleNamespace
from typing import Any

from zenml import step

from pioneerml.common.data_loader import LoaderFactory, BatchBundle
from pioneerml.common.pipeline.steps.step_types.model_runner.utils import build_loader_params
from pioneerml.common.pipeline.steps import BaseExportStep


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
        self.loader_factory = getattr(dataset, "loader_factory", None) or getattr(dataset, "loader", None)
        if not isinstance(self.loader_factory, LoaderFactory):
            raise RuntimeError("Dataset is missing a valid LoaderFactory instance.")
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
            loader = factory.build(config={"mode": "train", "batch_size": 1, "chunk_row_groups": 1, "chunk_workers": 0}).make_dataloader(shuffle_batches=False)
            for batch in loader:
                return (batch.x_node, batch.edge_index, batch.x_edge, batch.node_graph_id)
        return None

    def run(self) -> dict:
        cfg = self.config_json
        params = build_loader_params(
            cfg={
                "batch_size": 1,
                "chunk_row_groups": 1,
                "chunk_workers": 0,
                "loader_config": cfg.get("loader_config"),
            },
            purpose="export",
            forced_batch_size=1,
        )
        loader = self.loader_factory.build(config=params)
        data = loader.empty_data()
        data.source_main_sources = list(loader.input_sources.main_sources)

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
