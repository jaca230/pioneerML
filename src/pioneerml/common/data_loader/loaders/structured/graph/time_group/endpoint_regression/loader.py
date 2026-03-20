from __future__ import annotations

from typing import Any

import numpy as np
import torch

from pioneerml.common.data_loader.loaders.array_store import NDArrayColumnSpec
from pioneerml.common.data_loader.loaders.array_store.schemas import FeatureSchema, LoaderSchema, TargetSchema
from pioneerml.common.data_loader.loaders.config import DataFlowConfig, GraphTensorDims, SplitSampleConfig
from pioneerml.common.data_loader.loaders.factory.registry import REGISTRY as LOADER_REGISTRY
from pioneerml.common.data_loader.loaders.structured.graph.time_group.time_group_graph_loader import TimeGroupGraphLoader
from pioneerml.common.data_loader.loaders.stage.stages import (
    BaseStage,
    BatchPackStage,
    DistributedShardStage,
    EdgeFeatureStage,
    ExtractFeaturesStage,
    GraphLayoutStage,
    RowFilterStage,
    RowJoinStage,
)
from pioneerml.common.data_loader.loaders.input_source import (
    InputBackend,
    InputSourceSet,
    create_input_backend,
)
from pioneerml.common.staged_runtime.stage_observers import StageObserver

from .stages import (
    EndpointGraphFeatureStage,
    EndpointGraphTargetStage,
    EndpointNodeFeatureStage,
    EndpointQuantileTargetExpandStage,
)


@LOADER_REGISTRY.register("endpoint_regression")
class EndpointRegressionGraphLoader(TimeGroupGraphLoader):
    """Structured staged graph loader for endpoint regression."""

    NODE_FEATURE_DIM = 7
    EDGE_FEATURE_DIM = 4
    GRAPH_FEATURE_DIM = 4
    TARGET_DIM = 18
    NUM_CLASSES = TARGET_DIM

    def __init__(
        self,
        input_sources: InputSourceSet,
        *,
        mode: str = TimeGroupGraphLoader.MODE_TRAIN,
        input_backend: InputBackend | None = None,
        input_backend_name: str = "parquet",
        data_flow_config: DataFlowConfig | None = None,
        split_config: SplitSampleConfig | None = None,
        graph_dims: GraphTensorDims | None = None,
        stage_overrides: dict[str, BaseStage] | None = None,
        stage_observer: StageObserver | None = None,
        profiling: dict | None = None,
    ) -> None:
        self._resolved_field_specs: tuple[NDArrayColumnSpec, ...] = ()
        self.graph_dims = graph_dims or GraphTensorDims(
            node_feature_dim=int(self.NODE_FEATURE_DIM),
            edge_feature_dim=int(self.EDGE_FEATURE_DIM),
            graph_feature_dim=int(self.GRAPH_FEATURE_DIM),
            graph_target_dim=int(self.TARGET_DIM),
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
        features = FeatureSchema(
            fields=(
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
                NDArrayColumnSpec(
                    column="pred_hit_pion",
                    field="pred_hit_pion",
                    dtype=np.float32,
                    required=False,
                    target_only=False,
                ),
                NDArrayColumnSpec(
                    column="pred_hit_muon",
                    field="pred_hit_muon",
                    dtype=np.float32,
                    required=False,
                    target_only=False,
                ),
                NDArrayColumnSpec(
                    column="pred_hit_mip",
                    field="pred_hit_mip",
                    dtype=np.float32,
                    required=False,
                    target_only=False,
                ),
                NDArrayColumnSpec(
                    column="time_group_ids",
                    field="time_group_ids",
                    dtype=np.int64,
                    required=False,
                    target_only=False,
                ),
            )
        )
        targets = TargetSchema(
            fields=(
                NDArrayColumnSpec(column="group_start_x", field="group_start_x", dtype=np.float32, target_only=True),
                NDArrayColumnSpec(column="group_start_y", field="group_start_y", dtype=np.float32, target_only=True),
                NDArrayColumnSpec(column="group_start_z", field="group_start_z", dtype=np.float32, target_only=True),
                NDArrayColumnSpec(column="group_end_x", field="group_end_x", dtype=np.float32, target_only=True),
                NDArrayColumnSpec(column="group_end_y", field="group_end_y", dtype=np.float32, target_only=True),
                NDArrayColumnSpec(column="group_end_z", field="group_end_z", dtype=np.float32, target_only=True),
            )
        )
        return LoaderSchema(features=features, targets=targets)

    def default_stage_order(self) -> list[str]:
        return [
            "row_join",
            "row_filter",
            "distributed_shard",
            "extract_features",
            "build_layout",
            "build_nodes",
            "build_graph_features",
            "build_targets",
            "expand_quantile_targets",
            "build_edges",
            "pack_batch",
        ]

    def default_stages(self) -> dict[str, BaseStage]:
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
            "build_layout": GraphLayoutStage(
                row_group_count_fields=(
                    "group_start_x",
                    "group_start_y",
                    "group_start_z",
                    "group_end_x",
                    "group_end_y",
                    "group_end_z",
                ),
                input_state_key="features_in",
                source_state_keys=("features_in",),
                hits_time_group_field="hits_time_group",
            ),
            "build_nodes": EndpointNodeFeatureStage(
                input_state_key="features_in",
                coord_field="hits_coord",
                z_field="hits_z",
                edep_field="hits_edep",
                strip_type_field="hits_strip_type",
                time_group_field="hits_time_group",
                particle_mask_field="hits_particle_mask",
                node_feature_dim=int(self.empty_node_feature_dim()),
            ),
            "build_graph_features": EndpointGraphFeatureStage(
                input_state_key="features_in",
                graph_feature_dim=int(self.empty_graph_feature_dim()),
            ),
            "build_targets": EndpointGraphTargetStage(source_state_key="features_in"),
            "expand_quantile_targets": EndpointQuantileTargetExpandStage(repeats=3, target_key="y_graph"),
            "build_edges": EdgeFeatureStage(
                edge_feature_dim=int(self.empty_edge_feature_dim()),
                edge_populate_graph_block=None,
            ),
            "pack_batch": BatchPackStage(),
        }

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
        x_graph = batch.x_graph.to(device, non_blocking=(device.type == "cuda"))
        group_probs = x_graph[:, :3]
        u = x_graph[:, 3:4] if x_graph.shape[1] >= 4 else x.new_zeros((x_graph.shape[0], 1))
        splitter_probs = x.new_zeros((x.shape[0], 0))
        return (x, edge_index, edge_attr, node_graph_id, u, group_probs, splitter_probs), {}
