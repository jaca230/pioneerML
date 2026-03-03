from __future__ import annotations

from collections.abc import MutableMapping
from typing import Any

import pyarrow as pa

from pioneerml.common.loader.array_store.ndarray_store import NDArrayColumnSpec
from pioneerml.common.parquet import ParquetChunkReader, ParquetInputSet

from .base_stage import BaseStage


class RowJoinStage(BaseStage):
    """Row-level aligned join stage configured from parquet input sources."""

    name = "row_join"
    requires = ("table",)
    provides = ("table",)

    def __init__(
        self,
        *,
        parquet_inputs: ParquetInputSet | None = None,
        column_specs: tuple[NDArrayColumnSpec, ...] | None = None,
        source_columns_by_name: dict[str, tuple[str, ...]] | None = None,
        row_groups_per_chunk: int = 1,
    ) -> None:
        self.parquet_inputs = parquet_inputs
        if column_specs is not None:
            source_map: dict[str, list[str]] = {}
            for spec in column_specs:
                source = None if spec.source is None else str(spec.source)
                if source is None or source == "main":
                    continue
                source_map.setdefault(source, []).append(str(spec.column))
            self.source_columns_by_name = {
                str(k): tuple(dict.fromkeys(v))
                for k, v in source_map.items()
                if tuple(v)
            }
        else:
            self.source_columns_by_name = {
                str(k): tuple(str(c) for c in v)
                for k, v in dict(source_columns_by_name or {}).items()
                if tuple(v)
            }
        self.row_groups_per_chunk = max(1, int(row_groups_per_chunk))
        self._source_iters: dict[str, Any] = {}

    @staticmethod
    def _merge_columns(dst: pa.Table, src: pa.Table, columns: tuple[str, ...]) -> pa.Table:
        out = dst
        for col in columns:
            if col in out.column_names:
                out = out.set_column(out.schema.get_field_index(col), col, src.column(col))
            else:
                out = out.append_column(col, src.column(col))
        return out

    def _reset_source_iter(self, *, source_name: str) -> None:
        if self.parquet_inputs is None:
            self._source_iters[source_name] = None
            return
        source_paths = self.parquet_inputs.source_paths(source_name)
        source_cols = self.source_columns_by_name.get(source_name, ())
        if not source_paths or not source_cols:
            self._source_iters[source_name] = None
            return
        reader = ParquetChunkReader(
            parquet_paths=list(source_paths),
            columns=list(source_cols),
            row_groups_per_chunk=self.row_groups_per_chunk,
        )
        self._source_iters[source_name] = iter(reader.iter_tables())

    def join_table(self, *, table: pa.Table, state: MutableMapping[str, Any], loader) -> pa.Table | None:
        _ = state
        _ = loader
        merged = table
        for source_name, source_cols in self.source_columns_by_name.items():
            if int(state.get("chunk_index", 0)) == 0 or source_name not in self._source_iters:
                self._reset_source_iter(source_name=source_name)
            source_iter = self._source_iters.get(source_name)
            if source_iter is None:
                continue
            try:
                source_table = next(source_iter)
            except StopIteration as exc:
                raise RuntimeError(f"{source_name} chunk stream ended before main stream.") from exc

            if int(source_table.num_rows) != int(merged.num_rows):
                raise RuntimeError(
                    f"Aligned chunk row mismatch between main and {source_name} tables: "
                    f"{int(merged.num_rows)} vs {int(source_table.num_rows)}"
                )
            merged = self._merge_columns(merged, source_table, source_cols)
        return merged.combine_chunks()

    def run(self, *, state: MutableMapping[str, Any], loader) -> None:
        table = state.get("table")
        if table is None:
            state["chunk_out"] = None
            state["stop_pipeline"] = True
            return

        joined = self.join_table(table=table, state=state, loader=loader)
        if joined is None or int(joined.num_rows) == 0:
            state["table"] = None
            state["chunk_out"] = None
            state["stop_pipeline"] = True
            return
        state["table"] = joined
