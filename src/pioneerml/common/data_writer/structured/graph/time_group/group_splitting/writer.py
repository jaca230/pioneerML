from __future__ import annotations

import numpy as np
import pyarrow as pa

from pioneerml.common.data_writer.factory import register_writer
from pioneerml.common.data_writer.array_store import OutputColumnSpec, OutputSchema
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

from ..time_group_graph_data_writer import TimeGroupGraphDataWriter


@register_writer("group_splitting")
class GroupSplittingDataWriter(TimeGroupGraphDataWriter):
    """Writer implementation for group-splitting inference outputs."""

    def output_schema(self) -> OutputSchema:
        return OutputSchema(
            fields=(
                OutputColumnSpec("pred_hit_pion", model_output_name="main", output_index=0, dtype=np.float32, value_type=pa.float32()),
                OutputColumnSpec("pred_hit_muon", model_output_name="main", output_index=1, dtype=np.float32, value_type=pa.float32()),
                OutputColumnSpec("pred_hit_mip", model_output_name="main", output_index=2, dtype=np.float32, value_type=pa.float32()),
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
                    )
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
