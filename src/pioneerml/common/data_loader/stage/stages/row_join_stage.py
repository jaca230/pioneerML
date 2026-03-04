from __future__ import annotations

from collections.abc import MutableMapping
from typing import Any

import pyarrow as pa

from pioneerml.common.data_loader.array_store.ndarray_store import NDArrayColumnSpec
from pioneerml.common.data_loader.input_source import InputBackend, InputSourceSet

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
            return
        source_paths = input_sources.source_entries(source_name)
        source_fields = self.source_fields_by_name.get(source_name, ())
        if not source_paths or not source_fields:
            self._source_iters[source_name] = None
            return
        self._source_iters[source_name] = iter(
            input_backend.iter_tables(
                sources=source_paths,
                fields=list(source_fields),
                row_groups_per_chunk=self.row_groups_per_chunk,
            )
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
                source_table = next(source_iter)
            except StopIteration as exc:
                raise RuntimeError(f"{source_name} chunk stream ended before main stream.") from exc

            if int(source_table.num_rows) != int(merged.num_rows):
                raise RuntimeError(
                    f"Aligned chunk row mismatch between main and {source_name} tables: "
                    f"{int(merged.num_rows)} vs {int(source_table.num_rows)}"
                )
            merged = self._merge_fields(merged, source_table, source_fields)
        return merged.combine_chunks()

    def run_loader(self, *, state: MutableMapping[str, Any], owner) -> None:
        table = state.get("table")
        if table is None:
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
