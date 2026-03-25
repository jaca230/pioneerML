from __future__ import annotations

from collections.abc import MutableMapping
from typing import Any

import pyarrow as pa

from pioneerml.common.data_loader.loaders.array_store.ndarray_store import NDArrayColumnSpec
from pioneerml.common.data_loader.loaders.input_source import InputBackend, InputSourceSet

from .base_stage import BaseStage


class RowJoinStage(BaseStage):
    """Row-level aligned join stage configured from input sources."""

    name = "row_join"
    requires = ("table",)
    provides = ("table",)

    def __init__(
        self,
        *,
        input_sources: InputSourceSet | None = None,
        input_backend: InputBackend | None = None,
        field_specs: tuple[NDArrayColumnSpec, ...] | None = None,
        source_fields_by_name: dict[str, tuple[str, ...]] | None = None,
        row_groups_per_chunk: int = 1,
    ) -> None:
        self.input_sources = input_sources
        self.input_backend = input_backend
        if field_specs is not None:
            source_map: dict[str, list[str]] = {}
            for spec in field_specs:
                source = None if spec.source is None else str(spec.source)
                if source is None or source == "main":
                    continue
                source_map.setdefault(source, []).append(str(spec.column))
            self.source_fields_by_name = {
                str(k): tuple(dict.fromkeys(v))
                for k, v in source_map.items()
                if tuple(v)
            }
        else:
            self.source_fields_by_name = {
                str(k): tuple(str(c) for c in v)
                for k, v in dict(source_fields_by_name or {}).items()
                if tuple(v)
            }
        self.row_groups_per_chunk = max(1, int(row_groups_per_chunk))
        self._source_iters: dict[str, Any] = {}
        self._source_buffers: dict[str, pa.Table | None] = {}
        self._source_buffer_offsets: dict[str, int] = {}

    @staticmethod
    def _merge_fields(dst: pa.Table, src: pa.Table, fields: tuple[str, ...]) -> pa.Table:
        out = dst
        for field in fields:
            if field in out.column_names:
                out = out.set_column(out.schema.get_field_index(field), field, src.column(field))
            else:
                out = out.append_column(field, src.column(field))
        return out

    def _resolve_input_sources(self, *, owner) -> InputSourceSet | None:
        if self.input_sources is not None:
            return self.input_sources
        return getattr(owner, "input_sources", None)

    def _resolve_input_backend(self, *, owner) -> InputBackend | None:
        if self.input_backend is not None:
            return self.input_backend
        return getattr(owner, "input_backend", None)

    def _reset_source_iter(self, *, source_name: str, owner) -> None:
        input_sources = self._resolve_input_sources(owner=owner)
        input_backend = self._resolve_input_backend(owner=owner)
        if input_sources is None or input_backend is None:
            self._source_iters[source_name] = None
            self._source_buffers[source_name] = None
            self._source_buffer_offsets[source_name] = 0
            return
        source_paths = input_sources.source_entries(source_name)
        source_fields = self.source_fields_by_name.get(source_name, ())
        if not source_paths or not source_fields:
            self._source_iters[source_name] = None
            self._source_buffers[source_name] = None
            self._source_buffer_offsets[source_name] = 0
            return
        self._source_iters[source_name] = iter(
            input_backend.iter_tables(
                sources=source_paths,
                fields=list(source_fields),
                row_groups_per_chunk=self.row_groups_per_chunk,
            )
        )
        self._source_buffers[source_name] = None
        self._source_buffer_offsets[source_name] = 0

    @staticmethod
    def _buffer_remaining_rows(*, buffer: pa.Table | None, offset: int) -> int:
        if buffer is None:
            return 0
        return max(0, int(buffer.num_rows) - int(offset))

    def _next_aligned_slice(
        self,
        *,
        source_name: str,
        target_rows: int,
        owner,
    ) -> pa.Table:
        source_iter = self._source_iters.get(source_name)
        if source_iter is None:
            raise RuntimeError(f"{source_name} source iterator is not initialized.")

        buffer = self._source_buffers.get(source_name)
        offset = int(self._source_buffer_offsets.get(source_name, 0))
        need = max(0, int(target_rows))

        while self._buffer_remaining_rows(buffer=buffer, offset=offset) < need:
            try:
                next_table = next(source_iter)
            except StopIteration as exc:
                have = self._buffer_remaining_rows(buffer=buffer, offset=offset)
                raise RuntimeError(
                    f"{source_name} chunk stream ended before main stream "
                    f"(needed {need} rows, buffered {have})."
                ) from exc

            if int(next_table.num_rows) == 0:
                continue

            if buffer is None:
                buffer = next_table
                offset = 0
                continue

            if offset > 0:
                buffer = buffer.slice(offset)
                offset = 0
            buffer = pa.concat_tables([buffer, next_table], promote_options="none")

        if buffer is None:
            raise RuntimeError(f"{source_name} produced no rows while main stream has rows.")

        out = buffer.slice(offset, need)
        offset += need

        if offset >= int(buffer.num_rows):
            buffer = None
            offset = 0

        self._source_buffers[source_name] = buffer
        self._source_buffer_offsets[source_name] = int(offset)
        return out

    def _assert_sources_exhausted(self) -> None:
        for source_name, source_iter in self._source_iters.items():
            if source_iter is None:
                continue

            buffered = self._buffer_remaining_rows(
                buffer=self._source_buffers.get(source_name),
                offset=int(self._source_buffer_offsets.get(source_name, 0)),
            )
            if buffered > 0:
                raise RuntimeError(
                    f"Optional source '{source_name}' has {buffered} buffered rows left after main stream ended."
                )

            while True:
                try:
                    trailing = next(source_iter)
                except StopIteration:
                    break
                if int(trailing.num_rows) > 0:
                    raise RuntimeError(
                        f"Optional source '{source_name}' has trailing rows after main stream ended "
                        f"(example trailing chunk rows={int(trailing.num_rows)})."
                    )

    def join_table(self, *, table: pa.Table, state: MutableMapping[str, Any], owner) -> pa.Table | None:
        _ = state
        merged = table
        for source_name, source_fields in self.source_fields_by_name.items():
            if int(state.get("chunk_index", 0)) == 0 or source_name not in self._source_iters:
                self._reset_source_iter(source_name=source_name, owner=owner)
            source_iter = self._source_iters.get(source_name)
            if source_iter is None:
                continue
            try:
                source_table = self._next_aligned_slice(
                    source_name=source_name,
                    target_rows=int(merged.num_rows),
                    owner=owner,
                )
            except RuntimeError as exc:
                raise RuntimeError(
                    f"Aligned chunk row mismatch between main and {source_name} tables: {exc}"
                ) from exc
            merged = self._merge_fields(merged, source_table, source_fields)
        return merged.combine_chunks()

    def run_loader(self, *, state: MutableMapping[str, Any], owner) -> None:
        table = state.get("table")
        if table is None:
            self._assert_sources_exhausted()
            state["chunk_out"] = None
            state["stop_pipeline"] = True
            return

        joined = self.join_table(table=table, state=state, owner=owner)
        if joined is None or int(joined.num_rows) == 0:
            state["table"] = None
            state["chunk_out"] = None
            state["stop_pipeline"] = True
            return
        state["table"] = joined
