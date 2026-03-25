from __future__ import annotations

from typing import Any

import numpy as np
import torch

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
    EdgeFeatureStage,
    ExtractFeaturesStage,
    GraphLayoutStage,
    RowFilterStage,
    RowJoinStage,
)
from pioneerml.common.data_loader.loaders.structured.graph.time_group.time_group_graph_loader import TimeGroupGraphLoader
from pioneerml.common.staged_runtime.stage_observers import StageObserver

from .stages import (
    PositronAngleGraphFeatureStage,
    PositronAngleGraphTargetStage,
    PositronAngleNodeFeatureStage,
    PositronAngleQuantileTargetExpandStage,
    PositronAngleRelevantGraphFilterStage,
    PositronAngleSplitterFeatureStage,
)


@LOADER_REGISTRY.register("positron_angle")
class PositronAngleGraphLoader(TimeGroupGraphLoader):
    """Structured staged graph loader for positron-angle quantile regression."""

    NODE_FEATURE_DIM = 4
    EDGE_FEATURE_DIM = 4
    GRAPH_FEATURE_DIM = 3
    TARGET_DIM = 9
    NUM_CLASSES = 3
    ENDPOINT_DIM = 18
    EVENT_AFFINITY_DIM = 3
    PION_STOP_DIM = 3
    MIN_RELEVANT_HITS = 2
    POSITRON_PDG_ID = -11

    ENDPOINT_BASE_COLUMNS = (
        "pred_group_start_x",
        "pred_group_start_y",
        "pred_group_start_z",
        "pred_group_end_x",
        "pred_group_end_y",
        "pred_group_end_z",
    )
    ENDPOINT_QUANTILE_SUFFIXES = ("q16", "q50", "q84")
    PION_STOP_PRIOR_COLUMNS = (
        "pred_pion_stop_x_q50",
        "pred_pion_stop_y_q50",
        "pred_pion_stop_z_q50",
    )

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
            use_event_splitter_affinity=bool(params.get("use_event_splitter_affinity", True)),
            use_pion_stop_preds=bool(params.get("use_pion_stop_preds", True)),
            training_relevant_only=bool(params.get("training_relevant_only", True)),
            min_relevant_hits=int(params.get("min_relevant_hits", cls.MIN_RELEVANT_HITS)),
            stage_overrides=stage_overrides if isinstance(stage_overrides, dict) else None,
            stage_observer=stage_observer if isinstance(stage_observer, StageObserver) else None,
            profiling=profiling,
        )
        return cls._apply_common_loader_params(loader=loader, loader_params=params)

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
        use_group_probs: bool = True,
        use_splitter_probs: bool = True,
        use_endpoint_preds: bool = True,
        use_event_splitter_affinity: bool = True,
        use_pion_stop_preds: bool = True,
        training_relevant_only: bool = True,
        min_relevant_hits: int = MIN_RELEVANT_HITS,
        stage_overrides: dict[str, BaseStage] | None = None,
        stage_observer: StageObserver | None = None,
        profiling: dict | None = None,
    ) -> None:
        self._resolved_field_specs: tuple[NDArrayColumnSpec, ...] = ()
        self.use_group_probs = bool(use_group_probs)
        self.use_splitter_probs = bool(use_splitter_probs)
        self.use_endpoint_preds = bool(use_endpoint_preds)
        self.use_event_splitter_affinity = bool(use_event_splitter_affinity)
        self.use_pion_stop_preds = bool(use_pion_stop_preds)
        self.training_relevant_only = bool(training_relevant_only)
        self.min_relevant_hits = max(1, int(min_relevant_hits))

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
        features: list[NDArrayColumnSpec] = [
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
            NDArrayColumnSpec(
                column="hits_pdg_id",
                field="hits_pdg_id",
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
                column="edge_src_index",
                field="edge_src_index",
                dtype=np.int64,
                required=False,
                target_only=False,
            ),
            NDArrayColumnSpec(
                column="edge_dst_index",
                field="edge_dst_index",
                dtype=np.int64,
                required=False,
                target_only=False,
            ),
            NDArrayColumnSpec(
                column="pred_edge_affinity",
                field="pred_edge_affinity",
                dtype=np.float32,
                required=False,
                target_only=False,
            ),
        ]
        for col in self.endpoint_quantile_columns():
            features.append(
                NDArrayColumnSpec(column=col, field=col, dtype=np.float32, required=False, target_only=False)
            )
        for col in self.ENDPOINT_BASE_COLUMNS:
            features.append(
                NDArrayColumnSpec(column=col, field=col, dtype=np.float32, required=False, target_only=False)
            )
        for col in self.PION_STOP_PRIOR_COLUMNS:
            features.append(
                NDArrayColumnSpec(column=col, field=col, dtype=np.float32, required=False, target_only=False)
            )

        targets = TargetSchema(
            fields=(
                NDArrayColumnSpec(column="positron_px", field="positron_px", dtype=np.float32, target_only=True),
                NDArrayColumnSpec(column="positron_py", field="positron_py", dtype=np.float32, target_only=True),
                NDArrayColumnSpec(column="positron_pz", field="positron_pz", dtype=np.float32, target_only=True),
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
            "build_graph_features",
            "build_splitter_features",
            "build_targets",
            "expand_quantile_targets",
            "filter_relevant_graphs",
            "build_edges",
            "pack_batch",
        ]

    def default_stages(self) -> dict[str, BaseStage]:
        row_group_count_fields = (
            "positron_px",
            "positron_py",
            "positron_pz",
            "pred_pion",
            "pred_muon",
            "pred_mip",
            *self.endpoint_quantile_columns(),
            *self.ENDPOINT_BASE_COLUMNS,
            *self.PION_STOP_PRIOR_COLUMNS,
        )
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
                row_group_count_fields=row_group_count_fields,
                input_state_key="features_in",
                source_state_keys=("features_in",),
                hits_time_group_field="hits_time_group",
            ),
            "build_nodes": PositronAngleNodeFeatureStage(
                input_state_key="features_in",
                coord_field="hits_coord",
                z_field="hits_z",
                edep_field="hits_edep",
                strip_type_field="hits_strip_type",
                time_group_field="hits_time_group",
                particle_mask_field="hits_particle_mask",
                pdg_id_field="hits_pdg_id",
                node_feature_dim=int(self.empty_node_feature_dim()),
                num_classes=int(self.NUM_CLASSES),
            ),
            "build_graph_features": PositronAngleGraphFeatureStage(
                input_state_key="features_in",
                use_group_probs=self.use_group_probs,
                use_endpoint_preds=self.use_endpoint_preds,
                use_event_splitter_affinity=self.use_event_splitter_affinity,
                use_pion_stop_preds=self.use_pion_stop_preds,
                num_classes=int(self.NUM_CLASSES),
                endpoint_dim=int(self.ENDPOINT_DIM),
                event_affinity_dim=int(self.EVENT_AFFINITY_DIM),
                pion_stop_dim=int(self.PION_STOP_DIM),
            ),
            "build_splitter_features": PositronAngleSplitterFeatureStage(
                input_state_key="features_in",
                num_classes=int(self.NUM_CLASSES),
                use_splitter_probs=self.use_splitter_probs,
            ),
            "build_targets": PositronAngleGraphTargetStage(
                source_state_key="features_in",
            ),
            "expand_quantile_targets": PositronAngleQuantileTargetExpandStage(repeats=3, target_key="y_graph"),
            "filter_relevant_graphs": PositronAngleRelevantGraphFilterStage(
                training_relevant_only=self.training_relevant_only,
                min_relevant_hits=int(self.min_relevant_hits),
                positron_pdg_id=int(self.POSITRON_PDG_ID),
            ),
            "build_edges": EdgeFeatureStage(
                edge_feature_dim=int(self.empty_edge_feature_dim()),
                edge_populate_graph_block=None,
            ),
            "pack_batch": BatchPackStage(
                optional_tensor_state_fields={
                    "x_graph": "x_graph_out",
                    "y_graph": "y_graph",
                    "group_probs": "group_probs_out",
                    "splitter_probs": "splitter_probs_out",
                    "endpoint_preds": "endpoint_preds_out",
                    "event_affinity": "event_affinity_out",
                    "pion_stop_preds": "pion_stop_preds_out",
                },
            ),
        }

    def empty_data(self):
        data = super().empty_data()
        data.group_probs = torch.empty((0, int(self.NUM_CLASSES)), dtype=torch.float32)
        data.splitter_probs = torch.empty((0, int(self.NUM_CLASSES)), dtype=torch.float32)
        data.endpoint_preds = torch.empty((0, int(self.ENDPOINT_DIM)), dtype=torch.float32)
        data.event_affinity = torch.empty((0, int(self.EVENT_AFFINITY_DIM)), dtype=torch.float32)
        data.pion_stop_preds = torch.empty((0, int(self.PION_STOP_DIM)), dtype=torch.float32)
        data.event_ids = torch.empty((0,), dtype=torch.int64)
        data.group_ids = torch.empty((0,), dtype=torch.int64)
        data.num_groups = 0
        return data

    def _slice_chunk_batch(self, chunk: dict, g0: int, g1: int):
        d = super()._slice_chunk_batch(chunk, g0, g1)
        node_ptr = chunk["node_ptr"]
        n0 = int(node_ptr[g0].item())
        n1 = int(node_ptr[g1].item())

        d.group_probs = chunk["group_probs"][g0:g1]
        d.splitter_probs = chunk["splitter_probs"][n0:n1]
        d.endpoint_preds = chunk["endpoint_preds"][g0:g1]
        d.event_affinity = chunk["event_affinity"][g0:g1]
        d.pion_stop_preds = chunk["pion_stop_preds"][g0:g1]
        d.event_ids = d.graph_event_id
        d.group_ids = d.graph_time_group_id
        d.num_groups = int(d.num_graphs)
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
        group_probs = batch.group_probs.to(device, non_blocking=(device.type == "cuda"))
        splitter_probs = batch.splitter_probs.to(device, non_blocking=(device.type == "cuda"))
        endpoint_preds = batch.endpoint_preds.to(device, non_blocking=(device.type == "cuda"))
        event_affinity = batch.event_affinity.to(device, non_blocking=(device.type == "cuda"))
        pion_stop_preds = batch.pion_stop_preds.to(device, non_blocking=(device.type == "cuda"))
        return (
            x,
            edge_index,
            edge_attr,
            node_graph_id,
            group_probs,
            splitter_probs,
            endpoint_preds,
            event_affinity,
            pion_stop_preds,
        ), {}
