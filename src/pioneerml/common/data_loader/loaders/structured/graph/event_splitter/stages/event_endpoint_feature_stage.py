from __future__ import annotations

from collections.abc import MutableMapping
from typing import Any

import numpy as np

from pioneerml.common.data_loader.loaders.array_store import NDArrayStore
from pioneerml.common.data_loader.loaders.stage.stages import BaseStage


class EventEndpointFeatureStage(BaseStage):
    """Build per-group endpoint priors for event-splitter features."""

    name = "build_endpoint_features"
    requires = ("layout",)
    provides = ("endpoint_preds_out",)

    ENDPOINT_BASE_COLUMNS = (
        "pred_group_start_x",
        "pred_group_start_y",
        "pred_group_start_z",
        "pred_group_end_x",
        "pred_group_end_y",
        "pred_group_end_z",
    )
    ENDPOINT_QUANTILE_SUFFIXES = ("q16", "q50", "q84")

    def __init__(
        self,
        *,
        input_state_key: str = "features_in",
        use_endpoint_preds: bool = True,
        endpoint_dim: int = 18,
    ) -> None:
        self.input_state_key = str(input_state_key)
        self.use_endpoint_preds = bool(use_endpoint_preds)
        self.endpoint_dim = int(endpoint_dim)

    @classmethod
    def endpoint_quantile_columns(cls) -> tuple[str, ...]:
        out: list[str] = []
        for base in cls.ENDPOINT_BASE_COLUMNS:
            for suffix in cls.ENDPOINT_QUANTILE_SUFFIXES:
                out.append(f"{base}_{suffix}")
        return tuple(out)

    def run_loader(self, *, state: MutableMapping[str, Any], owner) -> None:
        _ = owner
        layout = state["layout"]
        chunk_in = state.get(self.input_state_key)
        if not isinstance(chunk_in, NDArrayStore):
            raise RuntimeError(
                f"Stage '{self.name}' missing required input state map: {self.input_state_key}"
            )

        total_groups = int(layout["total_groups"])
        row_group_counts = np.asarray(layout["row_group_counts"], dtype=np.int64)
        endpoint_preds_out = np.zeros((total_groups, self.endpoint_dim), dtype=np.float32)

        if not self.use_endpoint_preds or total_groups <= 0:
            state["endpoint_preds_out"] = endpoint_preds_out
            return

        quant_cols = self.endpoint_quantile_columns()
        has_quantile = all(
            chunk_in.has_raw(chunk_in.values_key(name)) and chunk_in.has_raw(chunk_in.offsets_key(name, 0))
            for name in quant_cols
        )
        has_base = all(
            chunk_in.has_raw(chunk_in.values_key(name)) and chunk_in.has_raw(chunk_in.offsets_key(name, 0))
            for name in self.ENDPOINT_BASE_COLUMNS
        )
        if not has_quantile and not has_base:
            state["endpoint_preds_out"] = endpoint_preds_out
            return

        if has_quantile:
            for base_idx, base_col in enumerate(self.ENDPOINT_BASE_COLUMNS):
                for q_idx, suffix in enumerate(self.ENDPOINT_QUANTILE_SUFFIXES):
                    name = f"{base_col}_{suffix}"
                    offs = chunk_in.offsets(name, 0)
                    counts = (offs[1:] - offs[:-1]).astype(np.int64, copy=False)
                    if not np.array_equal(counts, row_group_counts):
                        raise RuntimeError(f"{name} list lengths do not match inferred row group counts.")
                    vals = chunk_in.values(name).astype(np.float32, copy=False)
                    if int(vals.shape[0]) != total_groups:
                        raise RuntimeError(f"{name} flattened length must match total inferred groups.")
                    endpoint_preds_out[:, (base_idx * 3) + q_idx] = vals
        else:
            for base_idx, base_col in enumerate(self.ENDPOINT_BASE_COLUMNS):
                offs = chunk_in.offsets(base_col, 0)
                counts = (offs[1:] - offs[:-1]).astype(np.int64, copy=False)
                if not np.array_equal(counts, row_group_counts):
                    raise RuntimeError(f"{base_col} list lengths do not match inferred row group counts.")
                vals = chunk_in.values(base_col).astype(np.float32, copy=False)
                if int(vals.shape[0]) != total_groups:
                    raise RuntimeError(f"{base_col} flattened length must match total inferred groups.")
                endpoint_preds_out[:, (base_idx * 3) + 0] = vals
                endpoint_preds_out[:, (base_idx * 3) + 1] = vals
                endpoint_preds_out[:, (base_idx * 3) + 2] = vals

        state["endpoint_preds_out"] = endpoint_preds_out
