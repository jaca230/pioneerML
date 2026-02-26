from __future__ import annotations

import numpy as np

from pioneerml.common.loader.array_store.schemas import FeatureSchema, LoaderSchema, NDArrayField, TargetSchema
from pioneerml.common.loader.parquet.structured.graph.time_group.time_group_graph_loader import TimeGroupGraphLoader
from pioneerml.common.loader.stage.stages import (
    BaseStage,
    BatchPackStage,
    EdgeFeatureStage,
    ExtractFeaturesStage,
    NodeFeatureStage,
    RowFilterStage,
    RowJoinStage,
    TargetStage,
)


class GroupClassifierGraphLoader(TimeGroupGraphLoader):
    """Structured staged GroupClassifier graph loader."""

    _GRAPH_COLUMN_KEYS = (
        "event_id",
        "hits_time_group",
        "hits_strip_type",
        "hits_coord",
        "hits_z",
        "hits_edep",
    )
    _TARGET_COLUMN_KEYS = ("group_pion_in", "group_muon_in", "group_mip_in")
    _DEFAULT_COLUMN_MAP = {
        "event_id": "event_id",
        "hits_time_group": "hits_time_group",
        "hits_strip_type": "hits_strip_type",
        "hits_coord": "hits_coord",
        "hits_z": "hits_z",
        "hits_edep": "hits_edep",
        "group_pion_in": "group_pion_in",
        "group_muon_in": "group_muon_in",
        "group_mip_in": "group_mip_in",
    }

    NODE_FEATURE_DIM = 4
    EDGE_FEATURE_DIM = 4
    NUM_CLASSES = 3

    def __init__(
        self,
        parquet_paths: list[str],
        *,
        mode: str = TimeGroupGraphLoader.MODE_TRAIN,
        batch_size: int = 64,
        row_groups_per_chunk: int = 4,
        num_workers: int = 0,
        input_columns: list[str] | None = None,
        target_columns: list[str] | None = None,
        columns: list[str] | None = None,
        split: str | None = None,
        train_fraction: float = 0.9,
        val_fraction: float = 0.05,
        test_fraction: float = 0.05,
        split_seed: int = 0,
        sample_fraction: float | None = None,
        node_feature_dim: int = NODE_FEATURE_DIM,
        edge_feature_dim: int = EDGE_FEATURE_DIM,
        num_classes: int = NUM_CLASSES,
        column_map: dict[str, str] | None = None,
        stage_overrides: dict[str, BaseStage] | None = None,
        profiling: dict | None = None,
    ) -> None:
        self.column_map = dict(self._DEFAULT_COLUMN_MAP)
        if column_map:
            self.column_map.update({str(k): str(v) for k, v in dict(column_map).items()})

        inferred_input_columns = [self.column_map[k] for k in self._GRAPH_COLUMN_KEYS]
        inferred_target_columns = [self.column_map[k] for k in self._TARGET_COLUMN_KEYS]
        self.input_columns = list(inferred_input_columns if input_columns is None else input_columns)
        self.target_columns = list(inferred_target_columns if target_columns is None else target_columns)
        self.node_feature_dim = int(node_feature_dim)
        self.edge_feature_dim = int(edge_feature_dim)
        self.num_classes = int(num_classes)
        self.schema = self._build_schema()
        self.schema.validate_required_fields(
            [
                "event_id",
                "hits_time_group",
                "hits_coord",
                "hits_z",
                "hits_edep",
                "hits_strip_type",
                "group_pion_in",
                "group_muon_in",
                "group_mip_in",
            ],
            include_targets=True,
        )

        super().__init__(
            parquet_paths=parquet_paths,
            mode=mode,
            batch_size=batch_size,
            row_groups_per_chunk=row_groups_per_chunk,
            num_workers=num_workers,
            input_columns=self.input_columns,
            target_columns=self.target_columns,
            columns=columns,
            split=split,
            train_fraction=train_fraction,
            val_fraction=val_fraction,
            test_fraction=test_fraction,
            split_seed=split_seed,
            sample_fraction=sample_fraction,
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
                NDArrayField(name="event_id", column=self.column_map["event_id"], dtype=np.int64),
                NDArrayField(name="hits_time_group", column=self.column_map["hits_time_group"], dtype=np.int64),
                NDArrayField(name="hits_coord", column=self.column_map["hits_coord"], dtype=np.float32),
                NDArrayField(name="hits_z", column=self.column_map["hits_z"], dtype=np.float32),
                NDArrayField(name="hits_edep", column=self.column_map["hits_edep"], dtype=np.float32),
                NDArrayField(name="hits_strip_type", column=self.column_map["hits_strip_type"], dtype=np.int32),
            )
        )
        targets = TargetSchema(
            fields=(
                NDArrayField(name="group_pion_in", column=self.column_map["group_pion_in"], dtype=np.int32, target_only=True),
                NDArrayField(name="group_muon_in", column=self.column_map["group_muon_in"], dtype=np.int32, target_only=True),
                NDArrayField(name="group_mip_in", column=self.column_map["group_mip_in"], dtype=np.int32, target_only=True),
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
                event_id_column=self.column_map["event_id"],
                split=self.split,
                train_fraction=self.train_fraction,
                val_fraction=self.val_fraction,
                split_seed=self.split_seed,
                sample_fraction=self.sample_fraction,
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
                node_feature_dim=int(self.node_feature_dim),
            ),
            "build_targets": TargetStage(
                target_specs=(
                    ("group_pion_in", 0),
                    ("group_muon_in", 1),
                    ("group_mip_in", 2),
                ),
                num_classes=int(self.num_classes),
                source_state_key="features_in",
            ),
            "build_edges": EdgeFeatureStage(edge_feature_dim=int(self.edge_feature_dim), edge_populate_graph_block=None),
            "pack_batch": BatchPackStage(),
        }
