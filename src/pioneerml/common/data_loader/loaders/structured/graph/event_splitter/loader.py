from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np
import torch
from torch_geometric.data import Data

from pioneerml.common.data_loader.loaders.array_store import NDArrayColumnSpec
from pioneerml.common.data_loader.loaders.array_store.schemas import FeatureSchema, LoaderSchema, TargetSchema
from pioneerml.common.data_loader.loaders.config import DataFlowConfig, GraphTensorDims, SplitSampleConfig
from pioneerml.common.data_loader.loaders.factory.registry import REGISTRY as LOADER_REGISTRY
from pioneerml.common.data_loader.loaders.input_source import (
    InputBackend,
    InputSourceSet,
    create_input_backend,
)
from pioneerml.common.data_loader.loaders.stage.stages import (
    BaseStage,
    BatchPackStage,
    DistributedShardStage,
    ExtractFeaturesStage,
    RowFilterStage,
    RowJoinStage,
)
from pioneerml.common.data_loader.loaders.structured.graph.graph_loader import GraphLoader
from pioneerml.common.staged_runtime.stage_observers import StageObserver

from .stages import (
    EventEdgeFeatureStage,
    EventEdgeTargetStage,
    EventEndpointFeatureStage,
    EventGroupFeatureStage,
    EventLayoutStage,
    EventNodeFeatureStage,
    EventSplitterFeatureStage,
)


@LOADER_REGISTRY.register("event_splitter")
class EventSplitterGraphLoader(GraphLoader):
    """Structured staged graph loader for event-level edge-affinity splitting."""

    NODE_FEATURE_DIM = 4
    EDGE_FEATURE_DIM = 5
    NUM_CLASSES = 3
    ENDPOINT_DIM = 18

    ENDPOINT_BASE_COLUMNS = (
        "pred_group_start_x",
        "pred_group_start_y",
        "pred_group_start_z",
        "pred_group_end_x",
        "pred_group_end_y",
        "pred_group_end_z",
    )
    ENDPOINT_QUANTILE_SUFFIXES = ("q16", "q50", "q84")

    @classmethod
    def endpoint_quantile_columns(cls) -> tuple[str, ...]:
        out: list[str] = []
        for base in cls.ENDPOINT_BASE_COLUMNS:
            for suffix in cls.ENDPOINT_QUANTILE_SUFFIXES:
                out.append(f"{base}_{suffix}")
        return tuple(out)

    @classmethod
    def from_factory(
        cls,
        *,
        input_sources: InputSourceSet,
        input_backend_name: str,
        mode: str,
        data_flow_config: DataFlowConfig,
        split_config: SplitSampleConfig,
        loader_params: dict[str, Any] | None = None,
    ):
        params = dict(loader_params or {})
        stage_overrides = params.get("stage_overrides")
        stage_observer = params.get("stage_observer")
        profiling = dict(params.get("profiling") or {})
        loader = cls(
            input_sources=input_sources,
            mode=mode,
            data_flow_config=data_flow_config,
            split_config=split_config,
            input_backend=params.get("input_backend"),
            input_backend_name=input_backend_name,
            use_group_probs=bool(params.get("use_group_probs", True)),
            use_splitter_probs=bool(params.get("use_splitter_probs", True)),
            use_endpoint_preds=bool(params.get("use_endpoint_preds", True)),
            stage_overrides=stage_overrides if isinstance(stage_overrides, dict) else None,
            stage_observer=stage_observer if isinstance(stage_observer, StageObserver) else None,
            profiling=profiling,
        )
        return cls._apply_common_loader_params(loader=loader, loader_params=params)

    def __init__(
        self,
        input_sources: InputSourceSet,
        *,
        mode: str = GraphLoader.MODE_TRAIN,
        input_backend: InputBackend | None = None,
        input_backend_name: str = "parquet",
        data_flow_config: DataFlowConfig | None = None,
        split_config: SplitSampleConfig | None = None,
        graph_dims: GraphTensorDims | None = None,
        use_group_probs: bool = True,
        use_splitter_probs: bool = True,
        use_endpoint_preds: bool = True,
        stage_overrides: dict[str, BaseStage] | None = None,
        stage_observer: StageObserver | None = None,
        profiling: dict | None = None,
    ) -> None:
        self._resolved_field_specs: tuple[NDArrayColumnSpec, ...] = ()
        self.use_group_probs = bool(use_group_probs)
        self.use_splitter_probs = bool(use_splitter_probs)
        self.use_endpoint_preds = bool(use_endpoint_preds)
        self.graph_dims = graph_dims or GraphTensorDims(
            node_feature_dim=int(self.NODE_FEATURE_DIM),
            edge_feature_dim=int(self.EDGE_FEATURE_DIM),
            edge_target_dim=1,
        )
        self.schema = self.input_schema()

        include_targets = str(mode).strip().lower() != str(self.MODE_INFERENCE).lower()
        resolved_input_sources = input_sources
        resolved_input_backend = input_backend if input_backend is not None else create_input_backend(input_backend_name)
        declared_specs = self.schema.to_column_specs(include_targets=True)
        self._resolved_field_specs = resolved_input_backend.resolve_declared_field_specs(
            input_sources=resolved_input_sources,
            field_specs=declared_specs,
            include_targets=include_targets,
        )

        super().__init__(
            input_sources=resolved_input_sources,
            input_backend=resolved_input_backend,
            resolved_field_specs=self._resolved_field_specs,
            mode=mode,
            data_flow_config=data_flow_config,
            split_config=split_config,
            stage_overrides=stage_overrides,
            stage_observer=stage_observer,
            profiling=profiling,
        )

        required = self.required_fields(include_targets=self.include_targets)
        missing = [c for c in required if c not in self.main_fields]
        if missing:
            raise ValueError(f"Missing required columns for mode={self.mode}: {missing}")

    def input_schema(self) -> LoaderSchema:
        quant_cols = self.endpoint_quantile_columns()
        features = [
            NDArrayColumnSpec(column="event_id", field="event_id", dtype=np.int64, target_only=False),
            NDArrayColumnSpec(column="hits_time_group", field="hits_time_group", dtype=np.int64, target_only=False),
            NDArrayColumnSpec(column="hits_coord", field="hits_coord", dtype=np.float32, target_only=False),
            NDArrayColumnSpec(column="hits_z", field="hits_z", dtype=np.float32, target_only=False),
            NDArrayColumnSpec(column="hits_edep", field="hits_edep", dtype=np.float32, target_only=False),
            NDArrayColumnSpec(column="hits_strip_type", field="hits_strip_type", dtype=np.int32, target_only=False),
            NDArrayColumnSpec(
                column="hits_particle_mask",
                field="hits_particle_mask",
                dtype=np.int32,
                required=False,
                target_only=False,
            ),
            NDArrayColumnSpec(column="pred_pion", field="pred_pion", dtype=np.float32, required=False, target_only=False),
            NDArrayColumnSpec(column="pred_muon", field="pred_muon", dtype=np.float32, required=False, target_only=False),
            NDArrayColumnSpec(column="pred_mip", field="pred_mip", dtype=np.float32, required=False, target_only=False),
            NDArrayColumnSpec(column="pred_hit_pion", field="pred_hit_pion", dtype=np.float32, required=False, target_only=False),
            NDArrayColumnSpec(column="pred_hit_muon", field="pred_hit_muon", dtype=np.float32, required=False, target_only=False),
            NDArrayColumnSpec(column="pred_hit_mip", field="pred_hit_mip", dtype=np.float32, required=False, target_only=False),
        ]
        for name in quant_cols:
            features.append(
                NDArrayColumnSpec(column=name, field=name, dtype=np.float32, required=False, target_only=False)
            )
        for name in self.ENDPOINT_BASE_COLUMNS:
            features.append(
                NDArrayColumnSpec(column=name, field=name, dtype=np.float32, required=False, target_only=False)
            )
        targets = TargetSchema(
            fields=(
                NDArrayColumnSpec(
                    column="hits_contrib_mc_event_id",
                    field="hits_contrib_mc_event_id",
                    dtype=np.int32,
                    target_only=True,
                ),
            )
        )
        return LoaderSchema(features=FeatureSchema(fields=tuple(features)), targets=targets)

    def default_stage_order(self) -> list[str]:
        return [
            "row_join",
            "row_filter",
            "distributed_shard",
            "extract_features",
            "build_layout",
            "build_nodes",
            "build_group_features",
            "build_splitter_features",
            "build_endpoint_features",
            "build_edges",
            "build_edge_targets",
            "pack_batch",
        ]

    def default_stages(self) -> dict[str, BaseStage]:
        quant_cols = self.endpoint_quantile_columns()
        return {
            "row_join": RowJoinStage(
                input_sources=self.input_sources,
                input_backend=self.input_backend,
                field_specs=self._resolved_field_specs,
                row_groups_per_chunk=int(self.row_groups_per_chunk),
            ),
            "row_filter": RowFilterStage(
                event_id_column="event_id",
                split_config=self.split_config,
            ),
            "distributed_shard": DistributedShardStage(event_id_column="event_id"),
            "extract_features": ExtractFeaturesStage(
                column_specs=self.schema.to_column_specs(include_targets=True),
                output_state_key="features_in",
            ),
            "build_layout": EventLayoutStage(
                input_state_key="features_in",
                hits_time_group_field="hits_time_group",
                use_group_probs=self.use_group_probs,
                use_endpoint_preds=self.use_endpoint_preds,
                endpoint_quantile_columns=quant_cols,
                endpoint_base_columns=self.ENDPOINT_BASE_COLUMNS,
            ),
            "build_nodes": EventNodeFeatureStage(
                input_state_key="features_in",
                coord_field="hits_coord",
                z_field="hits_z",
                edep_field="hits_edep",
                strip_type_field="hits_strip_type",
                time_group_field="hits_time_group",
                particle_mask_field="hits_particle_mask",
                node_feature_dim=int(self.empty_node_feature_dim()),
                num_classes=int(self.NUM_CLASSES),
            ),
            "build_group_features": EventGroupFeatureStage(
                input_state_key="features_in",
                num_classes=int(self.NUM_CLASSES),
                use_group_probs=self.use_group_probs,
            ),
            "build_splitter_features": EventSplitterFeatureStage(
                input_state_key="features_in",
                num_classes=int(self.NUM_CLASSES),
                use_splitter_probs=self.use_splitter_probs,
            ),
            "build_endpoint_features": EventEndpointFeatureStage(
                input_state_key="features_in",
                use_endpoint_preds=self.use_endpoint_preds,
                endpoint_dim=int(self.ENDPOINT_DIM),
            ),
            "build_edges": EventEdgeFeatureStage(
                edge_feature_dim=int(self.empty_edge_feature_dim()),
                edge_populate_graph_block=None,
            ),
            "build_edge_targets": EventEdgeTargetStage(
                input_state_key="features_in",
                mc_event_field="hits_contrib_mc_event_id",
            ),
            "pack_batch": BatchPackStage(
                tensor_state_fields={
                    "x_node": "x_out",
                    "x_edge": "edge_attr_out",
                    "edge_index": "edge_index_out",
                    "graph_event_id": "graph_event_id",
                },
                tensor_layout_fields={
                    "node_ptr": "node_ptr",
                    "edge_ptr": "edge_ptr",
                },
                scalar_state_fields={"num_rows": "n_rows"},
                scalar_layout_fields={"num_graphs": "total_graphs"},
                optional_tensor_state_fields={
                    "y_edge": "y_edge",
                    "group_ptr": "group_ptr_out",
                    "time_group_ids": "time_group_ids_out",
                    "group_probs": "group_probs_out",
                    "splitter_probs": "splitter_probs_out",
                    "endpoint_preds": "endpoint_preds_out",
                },
            ),
        }

    def empty_data(self) -> Data:
        data = super().empty_data()
        data.group_ptr = torch.empty((0,), dtype=torch.int64)
        data.time_group_ids = torch.empty((0,), dtype=torch.int64)
        data.group_probs = torch.empty((0, int(self.NUM_CLASSES)), dtype=torch.float32)
        data.splitter_probs = torch.empty((0, int(self.NUM_CLASSES)), dtype=torch.float32)
        data.endpoint_preds = torch.empty((0, int(self.ENDPOINT_DIM)), dtype=torch.float32)
        data.graph_event_id = torch.empty((0,), dtype=torch.int64)
        data.num_groups = 0
        return data

    def _slice_chunk_batch(self, chunk: dict, g0: int, g1: int):
        d = super()._slice_chunk_batch(chunk, g0, g1)
        node_ptr = chunk["node_ptr"]
        group_ptr = chunk["group_ptr"]
        n0 = int(node_ptr[g0].item())
        n1 = int(node_ptr[g1].item())
        gp0 = int(group_ptr[g0].item())
        gp1 = int(group_ptr[g1].item())
        d.group_ptr = (group_ptr[g0 : g1 + 1] - gp0).to(dtype=torch.int64)
        d.time_group_ids = chunk["time_group_ids"][n0:n1]
        d.group_probs = chunk["group_probs"][gp0:gp1]
        d.splitter_probs = chunk["splitter_probs"][n0:n1]
        d.endpoint_preds = chunk["endpoint_preds"][gp0:gp1]
        d.num_groups = int(gp1 - gp0)
        return d

    def build_inference_model_input(
        self,
        *,
        batch,
        device: torch.device,
        cfg: dict[str, Any],
    ) -> tuple[tuple[Any, ...], dict[str, Any]]:
        _ = cfg
        x = batch.x_node.to(device, non_blocking=(device.type == "cuda"))
        edge_index = batch.edge_index.to(device, non_blocking=(device.type == "cuda"))
        edge_attr = batch.x_edge.to(device, non_blocking=(device.type == "cuda"))
        node_graph_id = batch.node_graph_id.to(device, non_blocking=(device.type == "cuda"))
        group_ptr = batch.group_ptr.to(device, non_blocking=(device.type == "cuda"))
        time_group_ids = batch.time_group_ids.to(device, non_blocking=(device.type == "cuda"))
        group_probs = batch.group_probs.to(device, non_blocking=(device.type == "cuda"))
        splitter_probs = batch.splitter_probs.to(device, non_blocking=(device.type == "cuda"))
        endpoint_preds = batch.endpoint_preds.to(device, non_blocking=(device.type == "cuda"))
        return (
            x,
            edge_index,
            edge_attr,
            node_graph_id,
            group_ptr,
            time_group_ids,
            group_probs,
            splitter_probs,
            endpoint_preds,
        ), {}
