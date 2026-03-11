from __future__ import annotations

import numpy as np

from pioneerml.common.data_loader.array_store import NDArrayColumnSpec
from pioneerml.common.data_loader.array_store.schemas import FeatureSchema, LoaderSchema, TargetSchema
from pioneerml.common.data_loader.config import DataFlowConfig, GraphTensorDims, SplitSampleConfig
from pioneerml.common.data_loader.factory import register_loader
from pioneerml.common.data_loader.structured.graph.time_group.time_group_graph_loader import TimeGroupGraphLoader
from pioneerml.common.data_loader.stage.stages import (
    BaseStage,
    BatchPackStage,
    DistributedShardStage,
    EdgeFeatureStage,
    ExtractFeaturesStage,
    GraphLayoutStage,
    NodeFeatureStage,
    RowFilterStage,
    RowJoinStage,
    GraphTargetStage,
)
from pioneerml.common.data_loader.input_source import (
    InputBackend,
    InputSourceSet,
    create_input_backend,
)


@register_loader("group_classifier")
class GroupClassifierGraphLoader(TimeGroupGraphLoader):
    """Structured staged GroupClassifier graph loader."""

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
        profiling: dict | None = None,
    ) -> None:
        self.graph_dims = graph_dims or GraphTensorDims(
            node_feature_dim=int(self.NODE_FEATURE_DIM),
            edge_feature_dim=int(self.EDGE_FEATURE_DIM),
            graph_target_dim=int(self.NUM_CLASSES),
        )
        self.schema = self.input_schema()

        resolved_input_sources = input_sources
        resolved_input_backend = input_backend if input_backend is not None else create_input_backend(input_backend_name)
        declared_specs = self.schema.to_column_specs(include_targets=True)
        self._resolved_field_specs = resolved_input_backend.resolve_declared_field_specs(
            input_sources=resolved_input_sources,
            field_specs=declared_specs,
            include_targets=(str(mode).strip().lower() != str(self.MODE_INFERENCE).lower()),
        )

        super().__init__(
            input_sources=resolved_input_sources,
            input_backend=resolved_input_backend,
            resolved_field_specs=self._resolved_field_specs,
            mode=mode,
            data_flow_config=data_flow_config,
            split_config=split_config,
            stage_overrides=stage_overrides,
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
            )
        )
        targets = TargetSchema(
            fields=(
                NDArrayColumnSpec(column="group_pion_in", field="group_pion_in", dtype=np.int32, target_only=True),
                NDArrayColumnSpec(column="group_muon_in", field="group_muon_in", dtype=np.int32, target_only=True),
                NDArrayColumnSpec(column="group_mip_in", field="group_mip_in", dtype=np.int32, target_only=True),
            )
        )
        return LoaderSchema(features=features, targets=targets)

    def default_stage_order(self) -> list[str]:
        return [
            "row_filter",
            "distributed_shard",
            "row_join",
            "extract_features",
            "build_layout",
            "build_nodes",
            "build_targets",
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
            "row_join": RowJoinStage(),
            "extract_features": ExtractFeaturesStage(
                column_specs=self.schema.to_column_specs(include_targets=True),
                output_state_key="features_in",
            ),
            "build_layout": GraphLayoutStage(
                row_group_count_fields=("group_pion_in", "group_muon_in", "group_mip_in"),
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
            "build_targets": GraphTargetStage(
                target_specs=(
                    ("group_pion_in", 0),
                    ("group_muon_in", 1),
                    ("group_mip_in", 2),
                ),
                num_classes=int(self.empty_graph_target_dim()),
                source_state_key="features_in",
            ),
            "build_edges": EdgeFeatureStage(
                edge_feature_dim=int(self.empty_edge_feature_dim()),
                edge_populate_graph_block=None,
            ),
            "pack_batch": BatchPackStage(),
        }
