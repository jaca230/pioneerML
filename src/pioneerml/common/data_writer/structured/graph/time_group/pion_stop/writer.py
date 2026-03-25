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

from ..time_group_graph_data_writer import TimeGroupGraphDataWriter


@WRITER_REGISTRY.register("pion_stop")
class PionStopDataWriter(TimeGroupGraphDataWriter):
    """Writer implementation for pion-stop quantile regression outputs."""

    _AXES = ("x", "y", "z")
    _QUANTILE_SUFFIXES = ("q16", "q50", "q84")

    @staticmethod
    def _quantile_tensor(preds: np.ndarray) -> np.ndarray:
        arr = np.asarray(preds, dtype=np.float32)
        if arr.ndim != 2:
            raise ValueError(f"Expected predictions to be 2D [N, D], got shape {tuple(arr.shape)}.")
        pred_dim = int(arr.shape[1])
        if pred_dim == 9:
            return arr.reshape(-1, 3, 3)
        if pred_dim == 3:
            return arr.reshape(-1, 3, 1).repeat(3, axis=2)
        raise ValueError(f"Unsupported pion-stop prediction dimension {pred_dim}. Expected 3 or 9.")

    def _slice_axis_quantile(self, *, axis_idx: int, quant_idx: int):
        def _extract(preds: np.ndarray) -> np.ndarray:
            q = self._quantile_tensor(preds)
            return q[:, int(axis_idx), int(quant_idx)]

        return _extract

    def output_schema(self) -> OutputSchema:
        fields: list[OutputColumnSpec] = []
        for axis_idx, axis_name in enumerate(self._AXES):
            base_name = f"pred_pion_stop_{axis_name}"
            fields.append(
                OutputColumnSpec(
                    base_name,
                    model_output_name="main",
                    transform=self._slice_axis_quantile(axis_idx=axis_idx, quant_idx=1),
                    dtype=np.float32,
                    value_type=pa.float32(),
                )
            )
            for q_idx, q_suffix in enumerate(self._QUANTILE_SUFFIXES):
                fields.append(
                    OutputColumnSpec(
                        f"{base_name}_{q_suffix}",
                        model_output_name="main",
                        transform=self._slice_axis_quantile(axis_idx=axis_idx, quant_idx=q_idx),
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
        preds = model_output[0] if isinstance(model_output, (tuple, list)) else model_output
        preds_np = preds.detach().cpu().to(torch.float32).numpy().astype("float32", copy=False)
        graph_event_ids_np = batch.graph_event_id.to(torch.int64).cpu().numpy().astype("int64", copy=False)
        graph_time_group_ids_np = batch.graph_time_group_id.to(torch.int64).cpu().numpy().astype("int64", copy=False)
        return TimeGroupPredictionSet(
            src_path=src_path,
            prediction_event_ids_np=graph_event_ids_np,
            model_outputs_by_name={"main": preds_np},
            num_rows=int(num_rows),
            time_group_event_ids_np=graph_event_ids_np,
            time_group_ids_np=graph_time_group_ids_np,
        )
