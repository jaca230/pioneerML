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
    NodeFeatureStage,
    RowFilterStage,
    RowJoinStage,
    GraphTargetStage,
)
from pioneerml.common.parquet import ParquetInputSet


class GroupClassifierGraphLoader(TimeGroupGraphLoader):
    """Structured staged GroupClassifier graph loader."""

    NODE_FEATURE_DIM = 4
    EDGE_FEATURE_DIM = 4
    NUM_CLASSES = 3

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
        self.graph_dims = graph_dims or GraphTensorDims(
            node_feature_dim=int(self.NODE_FEATURE_DIM),
            edge_feature_dim=int(self.EDGE_FEATURE_DIM),
            graph_target_dim=int(self.NUM_CLASSES),
        )
        self.schema = self._build_schema()

        declared_specs = self.schema.to_column_specs(include_targets=True)
        plan = self.resolve_declared_columns_from_sources(
            parquet_inputs=parquet_inputs,
            column_specs=declared_specs,
            include_targets=(str(mode).strip().lower() != str(self.MODE_INFERENCE).lower()),
        )
        self._resolved_column_specs = tuple(plan["resolved_column_specs"])

        super().__init__(
            parquet_inputs=parquet_inputs,
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
            "row_join": RowJoinStage(),
            "extract_features": ExtractFeaturesStage(
                column_specs=self.schema.to_column_specs(include_targets=True),
                output_state_key="features_in",
            ),
            "build_layout": self.make_time_group_layout_stage(
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
