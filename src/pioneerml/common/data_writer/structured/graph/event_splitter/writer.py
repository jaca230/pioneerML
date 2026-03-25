from __future__ import annotations

import numpy as np
import pyarrow as pa
import torch

from pioneerml.common.data_writer.array_store import OutputColumnSpec, OutputSchema
from pioneerml.common.data_writer.factory.registry import REGISTRY as WRITER_REGISTRY
from pioneerml.common.data_writer.input_source import TimeGroupPredictionSet
from pioneerml.common.data_writer.stage.stages import (
    AppendChunkStage,
    BufferChunkStage,
    CloseSinksStage,
    EmitRunOutputsStage,
    InitRunStateStage,
    OpenSinksStage,
    ResolveIndexingStage,
    ValidateInputsStage,
)
from pioneerml.common.data_writer.structured.graph.time_group.stages import (
    FinalizeBufferedWritesStage,
    StitchTimeGroupAlignedStructureStage,
)
from pioneerml.common.data_writer.structured.structured_data_writer import (
    WriterPhaseOrder,
    WriterPhaseStages,
)

from ..time_group.time_group_graph_data_writer import TimeGroupGraphDataWriter


@WRITER_REGISTRY.register("event_splitter")
class EventSplitterDataWriter(TimeGroupGraphDataWriter):
    """Writer implementation for event-splitter edge-affinity predictions."""

    def output_schema(self) -> OutputSchema:
        return OutputSchema(
            fields=(
                OutputColumnSpec(
                    "edge_src_index",
                    model_output_name="edge_src_index",
                    dtype=np.int64,
                    value_type=pa.int64(),
                ),
                OutputColumnSpec(
                    "edge_dst_index",
                    model_output_name="edge_dst_index",
                    dtype=np.int64,
                    value_type=pa.int64(),
                ),
                OutputColumnSpec(
                    "pred_edge_affinity",
                    model_output_name="pred_edge_affinity",
                    dtype=np.float32,
                    value_type=pa.float32(),
                ),
            )
        )

    def default_stage_order(self) -> WriterPhaseOrder:
        return WriterPhaseOrder(
            start=["init_run_state", "open_sinks"],
            chunk=[
                "validate_inputs",
                "resolve_indexing",
                "stitch_structure",
                "append_chunk",
                "buffer_chunk",
            ],
            finalize=["finalize_buffered_writes", "close_sinks", "emit_run_outputs"],
        )

    def default_stages(self) -> WriterPhaseStages:
        return WriterPhaseStages(
            start={
                "init_run_state": InitRunStateStage(),
                "open_sinks": OpenSinksStage(),
            },
            chunk={
                "validate_inputs": ValidateInputsStage(
                    required_state_keys=(
                        "src_path",
                        "prediction_event_ids_np",
                        "prediction_columns",
                        "num_rows",
                        "output_dir",
                        "time_group_event_ids_np",
                        "time_group_ids_np",
                    ),
                ),
                "resolve_indexing": ResolveIndexingStage(
                    index_keys=("prediction_event_ids_np", "time_group_event_ids_np", "time_group_ids_np")
                ),
                "stitch_structure": StitchTimeGroupAlignedStructureStage(
                    prediction_event_ids_key="prediction_event_ids_np",
                    time_group_event_ids_key="time_group_event_ids_np",
                    time_group_ids_key="time_group_ids_np",
                ),
                "append_chunk": AppendChunkStage(),
                "buffer_chunk": BufferChunkStage(),
            },
            finalize={
                "finalize_buffered_writes": FinalizeBufferedWritesStage(),
                "close_sinks": CloseSinksStage(),
                "emit_run_outputs": EmitRunOutputsStage(),
            },
        )

    def build_prediction_set(
        self,
        *,
        batch,
        model_output,
        src_path,
        num_rows: int,
        cfg: dict | None = None,
    ) -> TimeGroupPredictionSet:
        _ = cfg
        logits = model_output[0] if isinstance(model_output, (tuple, list)) else model_output
        probs_np = torch.sigmoid(logits).detach().cpu().to(torch.float32).reshape(-1).numpy().astype("float32", copy=False)

        edge_src_batch = batch.edge_index[0].to(torch.int64)
        edge_dst_batch = batch.edge_index[1].to(torch.int64)

        graph_event_ids = batch.graph_event_id.to(torch.int64)
        node_graph_id = batch.node_graph_id.to(torch.int64)
        edge_event_local = node_graph_id[edge_src_batch].to(torch.int64)
        node_ptr = batch.node_ptr.to(torch.int64)
        node_base = node_ptr[edge_event_local]
        edge_src_local_np = (edge_src_batch - node_base).cpu().numpy().astype("int64", copy=False)
        edge_dst_local_np = (edge_dst_batch - node_base).cpu().numpy().astype("int64", copy=False)
        edge_event_ids_np = graph_event_ids[edge_event_local].cpu().numpy().astype("int64", copy=False)

        node_event_ids_np = graph_event_ids[node_graph_id].cpu().numpy().astype("int64", copy=False)
        node_time_group_ids_np = batch.time_group_ids.to(torch.int64).cpu().numpy().astype("int64", copy=False)

        return TimeGroupPredictionSet(
            src_path=src_path,
            prediction_event_ids_np=edge_event_ids_np,
            model_outputs_by_name={
                "edge_src_index": edge_src_local_np,
                "edge_dst_index": edge_dst_local_np,
                "pred_edge_affinity": probs_np,
            },
            num_rows=int(num_rows),
            time_group_event_ids_np=node_event_ids_np,
            time_group_ids_np=node_time_group_ids_np,
        )
