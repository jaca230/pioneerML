from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq

from .base_output_backend import OutputBackend
from .factory.registry import REGISTRY as OUTPUT_BACKEND_REGISTRY


@dataclass
class _ParquetSink:
    dst_path: Path
    part_path: Path
    writer: pq.ParquetWriter | None = None
    buffered_tables: list[pa.Table] | None = None
    buffered_rows: int = 0


@OUTPUT_BACKEND_REGISTRY.register("parquet")
class ParquetOutputBackend(OutputBackend):
    def __init__(self, *, target_row_group_rows: int = 1024) -> None:
        rows = int(target_row_group_rows)
        if rows <= 0:
            raise ValueError("target_row_group_rows must be positive.")
        self.target_row_group_rows = rows

    def default_extension(self) -> str:
        return ".parquet"

    def write_table_atomic(self, *, table: pa.Table, dst_path: Path) -> None:
        dst_path.parent.mkdir(parents=True, exist_ok=True)
        part_path = dst_path.with_suffix(dst_path.suffix + ".part")
        if part_path.exists():
            part_path.unlink()
        pq.write_table(table, part_path, row_group_size=self.target_row_group_rows)
        os.replace(part_path, dst_path)

    def open_sink(self, *, dst_path: Path) -> Any:
        dst_path.parent.mkdir(parents=True, exist_ok=True)
        part_path = dst_path.with_suffix(dst_path.suffix + ".part")
        if part_path.exists():
            part_path.unlink()
        if dst_path.exists():
            dst_path.unlink()
        return _ParquetSink(dst_path=dst_path, part_path=part_path, writer=None, buffered_tables=[], buffered_rows=0)

    def _flush_buffer(self, *, sink: _ParquetSink) -> None:
        if int(sink.buffered_rows) <= 0:
            return
        tables = list(sink.buffered_tables or [])
        if not tables:
            sink.buffered_rows = 0
            return
        if len(tables) == 1:
            out_table = tables[0]
        else:
            out_table = pa.concat_tables(tables, promote_options="none")
        if sink.writer is None:
            sink.writer = pq.ParquetWriter(str(sink.part_path), out_table.schema)
        sink.writer.write_table(out_table, row_group_size=self.target_row_group_rows)
        sink.buffered_tables = []
        sink.buffered_rows = 0

    def append_chunk(self, *, sink: Any, table: pa.Table) -> None:
        if not isinstance(sink, _ParquetSink):
            raise TypeError(f"Expected _ParquetSink, got {type(sink).__name__}.")
        rows = int(table.num_rows)
        if rows <= 0:
            return
        buffered_tables = sink.buffered_tables
        if buffered_tables is None:
            buffered_tables = []
            sink.buffered_tables = buffered_tables
        buffered_tables.append(table)
        sink.buffered_rows += rows
        if int(sink.buffered_rows) >= int(self.target_row_group_rows):
            self._flush_buffer(sink=sink)

    def close_sink(self, *, sink: Any) -> None:
        if not isinstance(sink, _ParquetSink):
            raise TypeError(f"Expected _ParquetSink, got {type(sink).__name__}.")
        self._flush_buffer(sink=sink)
        if sink.writer is not None:
            sink.writer.close()
            os.replace(sink.part_path, sink.dst_path)
