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


@register_writer("endpoint_regression")
class EndpointRegressionDataWriter(TimeGroupGraphDataWriter):
    """Writer implementation for endpoint-regression inference outputs."""

    _POINT_NAMES = ("start", "end")
    _COORD_NAMES = ("x", "y", "z")
    _QUANTILE_SUFFIXES = ("q16", "q50", "q84")

    def _quantile_tensor(self, preds: np.ndarray) -> np.ndarray:
        arr = np.asarray(preds, dtype=np.float32)
        if arr.ndim != 2:
            raise ValueError(f"Expected predictions to be 2D [N, D], got shape {tuple(arr.shape)}.")
        pred_dim = int(arr.shape[1])
        if pred_dim == 18:
            return arr.reshape(-1, 2, 3, 3)
        if pred_dim == 6:
            return arr.reshape(-1, 2, 3, 1).repeat(3, axis=3)
        raise ValueError(f"Unsupported endpoint prediction dimension {pred_dim}. Expected 6 or 18.")

    def _slice_endpoint(self, *, point_idx: int, coord_idx: int, quant_idx: int):
        def _extract(preds: np.ndarray) -> np.ndarray:
            q = self._quantile_tensor(preds)
            return q[:, int(point_idx), int(coord_idx), int(quant_idx)]

        return _extract

    def output_schema(self) -> OutputSchema:
        fields: list[OutputColumnSpec] = []
        for point_idx, point_name in enumerate(self._POINT_NAMES):
            for coord_idx, coord_name in enumerate(self._COORD_NAMES):
                base_name = f"pred_group_{point_name}_{coord_name}"
                fields.append(
                    OutputColumnSpec(
                        base_name,
                        model_output_name="main",
                        transform=self._slice_endpoint(point_idx=point_idx, coord_idx=coord_idx, quant_idx=1),
                        dtype=np.float32,
                        value_type=pa.float32(),
                    )
                )
                for q_idx, q_suffix in enumerate(self._QUANTILE_SUFFIXES):
                    fields.append(
                        OutputColumnSpec(
                            f"{base_name}_{q_suffix}",
                            model_output_name="main",
                            transform=self._slice_endpoint(point_idx=point_idx, coord_idx=coord_idx, quant_idx=q_idx),
                            dtype=np.float32,
                            value_type=pa.float32(),
                        )
                    )
        return OutputSchema(fields=tuple(fields))

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
