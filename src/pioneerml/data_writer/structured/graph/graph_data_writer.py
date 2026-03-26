from __future__ import annotations

from pathlib import Path
from typing import Mapping

import numpy as np
import pyarrow as pa

from ..structured_data_writer import StructuredDataWriter


class GraphDataWriter(StructuredDataWriter):
    """Writer base for graph-structured outputs."""

    @staticmethod
    def normalize_prediction_columns(prediction_columns: Mapping[str, object]) -> dict[str, np.ndarray]:
        return {str(k): np.asarray(v) for k, v in dict(prediction_columns).items()}

    @staticmethod
    def validate_prediction_column_lengths(
        *,
        prediction_columns: Mapping[str, np.ndarray],
        expected_len: int,
    ) -> None:
        for name, values in prediction_columns.items():
            if int(values.shape[0]) != int(expected_len):
                raise ValueError(
                    f"Column '{name}' has leading dim {values.shape[0]} but expected {expected_len}."
                )

    @staticmethod
    def normalize_num_rows(event_ids_np: np.ndarray, num_rows: int) -> int:
        if int(num_rows) > 0:
            return int(num_rows)
        return (int(event_ids_np.max()) + 1) if event_ids_np.size > 0 else 0

    @staticmethod
    def validated_event_indices(*, event_ids_np: np.ndarray, num_rows: int) -> np.ndarray:
        return (event_ids_np >= 0) & (event_ids_np < int(num_rows))

    @staticmethod
    def stable_group_order(event_ids_np: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        if event_ids_np.size == 0:
            empty = np.zeros((0,), dtype=np.int64)
            return empty, empty
        order = np.argsort(event_ids_np, kind="stable")
        event_sorted = event_ids_np[order]
        return order, event_sorted

    @staticmethod
    def list_array_from_prediction_chunk(
        *,
        offsets: np.ndarray,
        values: np.ndarray,
        value_type: pa.DataType | None = None,
    ) -> pa.Array:
        if values.ndim < 1:
            raise ValueError(f"Prediction values must have at least 1 dimension, got shape {tuple(values.shape)}.")

        flat_values = pa.array(values.reshape(-1), type=value_type)
        value_array: pa.Array = flat_values
        for size in reversed(values.shape[1:]):
            value_array = pa.FixedSizeListArray.from_arrays(value_array, list_size=int(size))
        return pa.ListArray.from_arrays(offsets, value_array)

    def chunk_state(
        self,
        *,
        src_path: Path,
        prediction_event_ids_np: np.ndarray,
        prediction_columns: Mapping[str, np.ndarray],
        num_rows: int,
        output_dir: Path,
        output_path: str | None,
        write_timestamped: bool,
        timestamp: str,
        value_types: Mapping[str, pa.DataType] | None = None,
    ) -> dict[str, object]:
        columns = self.normalize_prediction_columns(prediction_columns)
        return {
            "src_path": src_path,
            "prediction_event_ids_np": np.asarray(prediction_event_ids_np, dtype=np.int64),
            "prediction_columns": columns,
            "num_rows": int(num_rows),
            "prediction_column_names": list(columns.keys()),
            "output_dir": output_dir,
            "output_path": output_path,
            "write_timestamped": bool(write_timestamped),
            "timestamp": str(timestamp),
            "streaming": bool(self.run_config.streaming),
            "value_types": value_types,
        }
