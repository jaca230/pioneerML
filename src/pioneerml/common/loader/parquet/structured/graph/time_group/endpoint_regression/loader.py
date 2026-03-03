from __future__ import annotations

import numpy as np

from pioneerml.common.loader.array_store import NDArrayColumnSpec
from pioneerml.common.loader.array_store.schemas import FeatureSchema, LoaderSchema, TargetSchema
from pioneerml.common.loader.config import DataFlowConfig, GraphTensorDims, SplitSampleConfig
from pioneerml.common.loader.parquet.structured.graph.time_group.time_group_graph_loader import TimeGroupGraphLoader
from pioneerml.common.loader.stage.stages import (
    BaseStage,
    BatchPackStage,
    EdgeFeatureStage,
    ExtractFeaturesStage,
    RowFilterStage,
    RowJoinStage,
)
from pioneerml.common.parquet import ParquetInputSet

from .stages import (
    EndpointGraphFeatureStage,
    EndpointGraphTargetStage,
    EndpointNodeFeatureStage,
    EndpointQuantileTargetExpandStage,
)


class EndpointRegressionGraphLoader(TimeGroupGraphLoader):
    """Structured staged graph loader for endpoint regression."""

    NODE_FEATURE_DIM = 7
    EDGE_FEATURE_DIM = 4
    GRAPH_FEATURE_DIM = 4
    TARGET_DIM = 18
    NUM_CLASSES = TARGET_DIM

    def __init__(
        self,
        parquet_inputs: ParquetInputSet,
        *,
        mode: str = TimeGroupGraphLoader.MODE_TRAIN,
        data_flow_config: DataFlowConfig | None = None,
        split_config: SplitSampleConfig | None = None,
        graph_dims: GraphTensorDims | None = None,
        stage_overrides: dict[str, BaseStage] | None = None,
        profiling: dict | None = None,
    ) -> None:
        self._resolved_column_specs: tuple[NDArrayColumnSpec, ...] = ()
        self.graph_dims = graph_dims or GraphTensorDims(
            node_feature_dim=int(self.NODE_FEATURE_DIM),
            edge_feature_dim=int(self.EDGE_FEATURE_DIM),
            graph_feature_dim=int(self.GRAPH_FEATURE_DIM),
            graph_target_dim=int(self.TARGET_DIM),
        )

        self.schema = self._build_schema()

        include_targets = str(mode).strip().lower() != str(self.MODE_INFERENCE).lower()
        parquet_input_set = parquet_inputs
        declared_specs = self.schema.to_column_specs(include_targets=True)
        plan = self.resolve_declared_columns_from_sources(
            parquet_inputs=parquet_input_set,
            column_specs=declared_specs,
            include_targets=include_targets,
        )
        self._resolved_column_specs = tuple(plan["resolved_column_specs"])

        super().__init__(
            parquet_inputs=parquet_input_set,
            resolved_column_specs=self._resolved_column_specs,
            mode=mode,
            data_flow_config=data_flow_config,
            split_config=split_config,
            stage_overrides=stage_overrides,
            profiling=profiling,
        )

        required = self.required_columns(include_targets=self.include_targets)
        missing = [c for c in required if c not in self.columns]
        if missing:
            raise ValueError(f"Missing required columns for mode={self.mode}: {missing}")

    def _build_schema(self) -> LoaderSchema:
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
                parquet_inputs=self.parquet_inputs,
                column_specs=self._resolved_column_specs,
                row_groups_per_chunk=int(self.row_groups_per_chunk),
            ),
            "row_filter": RowFilterStage(
                event_id_column="event_id",
                split_config=self.split_config,
            ),
            "extract_features": ExtractFeaturesStage(
                column_specs=self.schema.to_column_specs(include_targets=True),
                output_state_key="features_in",
            ),
            "build_layout": self.make_time_group_layout_stage(
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
