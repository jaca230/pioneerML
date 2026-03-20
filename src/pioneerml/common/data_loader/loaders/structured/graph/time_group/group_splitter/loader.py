from __future__ import annotations

import numpy as np

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
    NodeFeatureStage,
    RowFilterStage,
    RowJoinStage,
)
from pioneerml.common.data_loader.loaders.input_source import (
    InputBackend,
    InputSourceSet,
    create_input_backend,
)
from pioneerml.common.staged_runtime.stage_observers import StageObserver

from .stages import GroupFeatureStage, NodeTargetStage


@LOADER_REGISTRY.register("group_splitter")
class GroupSplitterGraphLoader(TimeGroupGraphLoader):
    """Structured staged GroupSplitter graph loader."""

    NODE_FEATURE_DIM = 4
    EDGE_FEATURE_DIM = 4
    NUM_CLASSES = 3

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
            node_target_dim=int(self.NUM_CLASSES),
            graph_feature_dim=int(self.NUM_CLASSES),
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
                    column="pred_pion",
                    field="pred_pion",
                    dtype=np.float32,
                    required=False,
                    target_only=False,
                ),
                NDArrayColumnSpec(
                    column="pred_muon",
                    field="pred_muon",
                    dtype=np.float32,
                    required=False,
                    target_only=False,
                ),
                NDArrayColumnSpec(
                    column="pred_mip",
                    field="pred_mip",
                    dtype=np.float32,
                    required=False,
                    target_only=False,
                ),
            )
        )
        targets = TargetSchema(
            fields=(
                NDArrayColumnSpec(
                    column="hits_particle_mask",
                    field="hits_particle_mask",
                    dtype=np.int32,
                    target_only=True,
                ),
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
            "build_node_targets",
            "build_graph_features",
            "build_edges",
            "pack_batch",
        ]

    def default_stages(self) -> dict[str, BaseStage]:
        return {
            "row_filter": RowFilterStage(
                event_id_column="event_id",
                split_config=self.split_config,
            ),
            "distributed_shard": DistributedShardStage(event_id_column="event_id"),
            "row_join": RowJoinStage(
                input_sources=self.input_sources,
                input_backend=self.input_backend,
                field_specs=self._resolved_field_specs,
                row_groups_per_chunk=int(self.row_groups_per_chunk),
            ),
            "extract_features": ExtractFeaturesStage(
                column_specs=self.schema.to_column_specs(include_targets=True),
                output_state_key="features_in",
            ),
            "build_layout": GraphLayoutStage(
                row_group_count_fields=(),
                input_state_key="features_in",
                source_state_keys=("features_in",),
                hits_time_group_field="hits_time_group",
            ),
            "build_nodes": NodeFeatureStage(
                input_state_key="features_in",
                coord_field="hits_coord",
                z_field="hits_z",
                edep_field="hits_edep",
                strip_type_field="hits_strip_type",
                time_group_field="hits_time_group",
                node_feature_dim=int(self.empty_node_feature_dim()),
            ),
            "build_node_targets": NodeTargetStage(
                input_state_key="features_in",
                particle_mask_field="hits_particle_mask",
                num_classes=int(self.empty_node_target_dim()),
            ),
            "build_graph_features": GroupFeatureStage(
                input_state_key="features_in",
                num_classes=int(self.empty_graph_feature_dim()),
            ),
            "build_edges": EdgeFeatureStage(
                edge_feature_dim=int(self.empty_edge_feature_dim()),
                edge_populate_graph_block=None,
            ),
            "pack_batch": BatchPackStage(),
        }
