from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Callable

import numpy as np
import pyarrow as pa


OutputTransform = Callable[[np.ndarray], np.ndarray]


@dataclass(frozen=True)
class OutputColumnSpec:
    """Declarative spec for PredictionSet model output -> written column."""

    output_column: str
    model_output_name: str = "main"
    output_index: int | None = None
    dtype: Any | None = None
    value_type: pa.DataType | None = None
    required: bool = True
    transform: OutputTransform | None = None


@dataclass(frozen=True)
class OutputSchema:
    fields: tuple[OutputColumnSpec, ...]

    def column_names(self) -> tuple[str, ...]:
        return tuple(str(f.output_column) for f in self.fields)

    def value_types(self) -> dict[str, pa.DataType]:
        out: dict[str, pa.DataType] = {}
        for f in self.fields:
            if f.value_type is not None:
                out[str(f.output_column)] = f.value_type
        return out

    def prediction_columns(self, *, model_outputs_by_name: Mapping[str, np.ndarray]) -> dict[str, np.ndarray]:
        outputs = {str(k): np.asarray(v) for k, v in dict(model_outputs_by_name).items()}
        out: dict[str, np.ndarray] = {}
        for spec in self.fields:
            output_name = str(spec.model_output_name)
            values = outputs.get(output_name)
            if values is None:
                if bool(spec.required):
                    raise ValueError(f"Missing required model output '{output_name}' for column '{spec.output_column}'.")
                continue

            col = values
            if spec.transform is not None:
                col = np.asarray(spec.transform(values))
            elif spec.output_index is not None:
                if col.ndim < 2:
                    raise ValueError(
                        f"Model output '{output_name}' must be 2D+ for output_index mapping "
                        f"(column '{spec.output_column}'). Got shape {tuple(col.shape)}."
                    )
                col = col[:, int(spec.output_index)]

            if spec.dtype is not None:
                col = col.astype(spec.dtype, copy=False)
            out[str(spec.output_column)] = np.asarray(col)
        return out

