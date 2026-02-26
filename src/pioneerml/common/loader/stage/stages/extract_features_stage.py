from __future__ import annotations

from collections.abc import Callable, MutableMapping, Sequence
from typing import Any

import numpy as np
import pyarrow as pa

from ...array_store.ndarray_store import NDArrayColumnSpec, NDArrayStore
from .base_stage import BaseStage

CustomExtractor = Callable[[pa.Table, Any, MutableMapping[str, Any]], dict[str, np.ndarray] | NDArrayStore]


class ExtractFeaturesStage(BaseStage):
    """General Arrow->ndarray extraction stage with automatic nesting support."""

    name = "extract_features"
    requires = ("table",)
    provides = ("chunk_in", "n_rows")

    def __init__(
        self,
        *,
        column_specs: Sequence[NDArrayColumnSpec] | None = None,
        custom_extractors: Sequence[CustomExtractor] | None = None,
        output_state_key: str = "chunk_in",
        write_n_rows: bool = True,
        only_if_include_targets: bool = False,
    ) -> None:
        self._column_specs = tuple(column_specs or ())
        self._custom_extractors = tuple(custom_extractors or ())
        self.output_state_key = str(output_state_key)
        self.write_n_rows = bool(write_n_rows)
        self.only_if_include_targets = bool(only_if_include_targets)

    def get_column_specs(self, *, loader, state: MutableMapping[str, Any]) -> Sequence[NDArrayColumnSpec]:
        _ = state
        include_targets = bool(getattr(loader, "include_targets", False))
        if include_targets:
            return self._column_specs
        return tuple(spec for spec in self._column_specs if not bool(spec.target_only))

    def get_custom_extractors(self, *, loader, state: MutableMapping[str, Any]) -> Sequence[CustomExtractor]:
        _ = loader
        _ = state
        return self._custom_extractors

    @staticmethod
    def _to_np(arr: pa.Array, dtype=None) -> np.ndarray:
        out = arr.to_numpy(zero_copy_only=False)
        if dtype is not None and out.dtype != dtype:
            out = out.astype(dtype, copy=False)
        return out

    @classmethod
    def _extract_ndarrays(cls, *, arr: pa.Array, dtype=None) -> tuple[np.ndarray, list[np.ndarray], list[int]]:
        offsets: list[np.ndarray] = []
        fixed_shape: list[int] = []
        leaf = arr

        while pa.types.is_list(leaf.type) or pa.types.is_large_list(leaf.type):
            offsets.append(cls._to_np(leaf.offsets, np.int64))
            leaf = leaf.values

        while pa.types.is_fixed_size_list(leaf.type):
            fixed_shape.append(int(leaf.type.list_size))
            leaf = leaf.values

        values = cls._to_np(leaf, dtype)
        if fixed_shape:
            values = values.reshape((-1, *fixed_shape))

        return values, offsets, fixed_shape

    @classmethod
    def _extract_column(cls, *, table: pa.Table, spec: NDArrayColumnSpec, store: NDArrayStore) -> None:
        if spec.column not in table.column_names:
            if spec.required:
                raise RuntimeError(f"Missing required column: {spec.column}")
            return

        arr = table.column(spec.column).chunk(0)
        values, offsets, fixed_shape = cls._extract_ndarrays(arr=arr, dtype=spec.dtype)
        store.set_values(spec.field, values)
        for i, off in enumerate(offsets):
            store.set_offsets(spec.field, i, off)
        if fixed_shape:
            store.set_shape(spec.field, np.asarray(fixed_shape, dtype=np.int64))
        if spec.include_validity:
            store.set_valid(spec.field, arr.is_valid().to_numpy(zero_copy_only=False))

    def run(self, *, state: MutableMapping[str, Any], loader) -> None:
        table = state.get("table")
        if table is None or int(table.num_rows) == 0:
            state["chunk_out"] = None
            state["stop_pipeline"] = True
            return

        if self.only_if_include_targets and not bool(getattr(loader, "include_targets", False)):
            state[self.output_state_key] = NDArrayStore()
            return

        n_rows = int(table.num_rows)
        out_store = NDArrayStore()

        for spec in self.get_column_specs(loader=loader, state=state):
            self._extract_column(table=table, spec=spec, store=out_store)

        for extractor in self.get_custom_extractors(loader=loader, state=state):
            extra = extractor(table, loader, state)
            if extra:
                if isinstance(extra, NDArrayStore):
                    out_store.update(extra.raw())
                else:
                    out_store.update(extra)

        if self.write_n_rows:
            state["n_rows"] = n_rows
        state[self.output_state_key] = out_store
