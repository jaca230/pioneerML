from __future__ import annotations

from pioneerml.common.pipeline.services import BaseExportService
from pioneerml.pipelines.training.event_splitting.dataset import EventSplitterDataset

from ..base import EventSplitterServiceBase


class EventSplitterExportService(EventSplitterServiceBase, BaseExportService):
    step_key = "export"

    def __init__(
        self,
        *,
        dataset,
        module,
        pipeline_config: dict | None = None,
        hpo_params: dict | None = None,
        metrics: dict | None = None,
    ) -> None:
        super().__init__(dataset=dataset, pipeline_config=pipeline_config)
        self.module = module
        self.hpo_params = hpo_params
        self.metrics = metrics

    def default_config(self) -> dict:
        return {}

    @staticmethod
    def _build_export_example(dataset: EventSplitterDataset):
        data = dataset.data
        if hasattr(data, "batch"):
            return (
                data.x,
                data.edge_index,
                data.edge_attr,
                data.batch,
                data.group_ptr,
                data.time_group_ids,
                data.group_probs,
                data.splitter_probs,
                data.endpoint_preds,
            )
        factory = getattr(dataset, "loader_factory", None) or getattr(dataset, "loader", None)
        if factory is not None:
            loader = factory.build_loader(
                loader_params={
                    "mode": "train",
                    "use_group_probs": True,
                    "use_splitter_probs": True,
                    "use_endpoint_preds": True,
                    "batch_size": 1,
                    "chunk_row_groups": 1,
                    "chunk_workers": 0,
                }
            ).make_dataloader(shuffle_batches=False)
            for batch in loader:
                return (
                    batch.x,
                    batch.edge_index,
                    batch.edge_attr,
                    batch.batch,
                    batch.group_ptr,
                    batch.time_group_ids,
                    batch.group_probs,
                    batch.splitter_probs,
                    batch.endpoint_preds,
                )
        return None

    def execute(self) -> dict:
        cfg = self.get_config()
        params = self._resolve_loader_params(
            {
                "batch_size": 1,
                "chunk_row_groups": 1,
                "chunk_workers": 0,
                "use_group_probs": True,
                "use_splitter_probs": True,
                "use_endpoint_preds": True,
                "loader_config": cfg.get("loader_config"),
            },
            purpose="export",
            forced_batch_size=1,
        )
        loader = self.loader_factory.build_loader(loader_params=params)
        data, targets = loader.empty_data()
        data.source_parquet_paths = list(loader.parquet_paths)
        if loader.group_probs_parquet_paths is not None:
            data.group_probs_parquet_paths = list(loader.group_probs_parquet_paths)
        if loader.group_splitter_parquet_paths is not None:
            data.group_splitter_parquet_paths = list(loader.group_splitter_parquet_paths)
        if loader.endpoint_parquet_paths is not None:
            data.endpoint_parquet_paths = list(loader.endpoint_parquet_paths)
        dataset = EventSplitterDataset(
            data=data,
            targets=targets,
            loader_factory=self.loader_factory,
            loader=self.loader_factory,
        )
        return self.export_torchscript(
            module=self.module,
            dataset=dataset,
            cfg=cfg,
            pipeline_config=self.pipeline_config,
            hpo_params=self.hpo_params,
            metrics=self.metrics,
            default_export_dir="trained_models/event_splitter",
            default_prefix="event_splitter",
            example_builder=self._build_export_example,
        )
