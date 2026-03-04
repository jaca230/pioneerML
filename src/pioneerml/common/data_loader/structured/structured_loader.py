from __future__ import annotations

from abc import abstractmethod
from collections.abc import Iterator, MutableMapping
from typing import Any

import torch

from pioneerml.common.data_loader.array_store.ndarray_store import NDArrayColumnSpec
from pioneerml.common.data_loader.array_store.schemas import LoaderSchema
from pioneerml.common.data_loader.base_loader import BaseLoader
from pioneerml.common.data_loader.config import DataFlowConfig, SplitSampleConfig
from pioneerml.common.data_loader.input_source import InputBackend, InputSourceSet, ParquetInputBackend
from pioneerml.common.data_loader.stage.loader_diagnostics import LoaderDiagnostics
from pioneerml.common.data_loader.stage.loader_stage_context import LoaderStageContext
from pioneerml.common.staged_runtime import PhaseRunner, StageRunner
from pioneerml.common.staged_runtime.stage_observers import (
    CompositeStageObserver,
    JsonlObserver,
    MemoryObserver,
    StageObserver,
    TimingObserver,
)
from pioneerml.common.data_loader.stage.stages import BaseStage


class StructuredLoader(BaseLoader):
    """Structured staged loader using an input backend contract."""

    def __init__(
        self,
        *,
        input_sources: InputSourceSet,
        resolved_field_specs: tuple[NDArrayColumnSpec, ...] | None = None,
        mode: str | None = None,
        data_flow_config: DataFlowConfig | None = None,
        split_config: SplitSampleConfig | None = None,
        input_backend: InputBackend | None = None,
        stage_overrides: dict[str, BaseStage] | None = None,
        stage_observer: StageObserver | None = None,
        profiling: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(data_flow_config=data_flow_config, mode=mode)
        self.input_sources = input_sources
        self.input_backend = input_backend if input_backend is not None else ParquetInputBackend()
        self.resolved_field_specs = tuple(resolved_field_specs if resolved_field_specs is not None else ())

        self.input_fields = self.input_backend.fields_from_specs(
            field_specs=self.resolved_field_specs,
            target_only=False,
            required=True,
        )
        self.target_fields = self.input_backend.fields_from_specs(
            field_specs=self.resolved_field_specs,
            target_only=True,
            required=True,
        )
        self.optional_input_fields = self.input_backend.fields_from_specs(
            field_specs=self.resolved_field_specs,
            target_only=False,
            required=False,
        )
        self.main_fields = self.input_backend.fields_from_specs(
            field_specs=self.resolved_field_specs,
            source="main",
        )

        self.row_groups_per_chunk = int(self.data_flow_config.row_groups_per_chunk)
        self.split_config = split_config if split_config is not None else SplitSampleConfig()
        self.split = self.split_config.split
        self.train_fraction = float(self.split_config.train_fraction)
        self.val_fraction = float(self.split_config.val_fraction)
        self.test_fraction = float(self.split_config.test_fraction)
        self.split_seed = int(self.split_config.split_seed if self.split_config.split_seed is not None else 0)
        self.sample_fraction = self.split_config.sample_fraction
        self.edge_populate_graph_block = 512

        self.stage_overrides = dict(stage_overrides or {})
        self.profiling = dict(profiling or {})
        self.diagnostics = LoaderDiagnostics(loader_kind=self.__class__.__name__)
        self.stage_sequence = self._build_stage_sequence()
        self.phase_runner = PhaseRunner()
        self.phase_runner.register_phase(
            name="load_chunk",
            runner=StageRunner(
            stages=self.stage_sequence,
            observer=self._build_stage_observer(stage_observer=stage_observer),
            context_cls=LoaderStageContext,
            ),
        )

    def required_fields(self, *, include_targets: bool | None = None) -> list[str]:
        use_targets = self.include_targets if include_targets is None else bool(include_targets)
        cols = [*self.input_fields]
        if use_targets:
            cols.extend(self.target_fields)
        return list(dict.fromkeys(cols))

    def _iter_tables(self):
        yield from self.input_backend.iter_tables(
            sources=self.input_sources.main_sources,
            fields=list(self.main_fields),
            row_groups_per_chunk=int(self.row_groups_per_chunk),
        )

    @abstractmethod
    def default_stage_order(self) -> list[str]:
        raise NotImplementedError

    @abstractmethod
    def default_stages(self) -> dict[str, BaseStage]:
        raise NotImplementedError

    @abstractmethod
    def input_schema(self) -> LoaderSchema:
        raise NotImplementedError

    def _build_stage_sequence(self) -> list[BaseStage]:
        stages = dict(self.default_stages())
        stages.update(self.stage_overrides)
        order = self.default_stage_order()
        missing = [name for name in order if name not in stages]
        if missing:
            raise RuntimeError(f"Missing stage implementations for: {missing}")
        return [stages[name] for name in order]

    def _build_stage_observer(self, *, stage_observer: StageObserver | None) -> StageObserver:
        if stage_observer is not None:
            return stage_observer

        observers: list[StageObserver] = [TimingObserver(diagnostics=self.diagnostics)]
        if bool(self.profiling.get("memory", False)):
            observers.append(
                MemoryObserver(
                    diagnostics=self.diagnostics,
                    track_rss=bool(self.profiling.get("rss", True)),
                    track_vram=bool(self.profiling.get("vram", False)),
                )
            )
        jsonl_path = self.profiling.get("jsonl_path")
        if jsonl_path:
            observers.append(JsonlObserver(path=str(jsonl_path), append=bool(self.profiling.get("jsonl_append", True))))
        if len(observers) == 1:
            return observers[0]
        return CompositeStageObserver(observers)

    def _run_stage_sequence(self, *, state: MutableMapping[str, Any]) -> MutableMapping[str, Any]:
        chunk_index = int(state.get("chunk_index", 0))
        raw_num_rows = int(state.get("raw_num_rows", 0))
        return self.phase_runner.run_phase(
            name="load_chunk",
            state=state,
            owner=self,
            context_fields={
                "chunk_index": chunk_index,
                "raw_num_rows": raw_num_rows,
            },
        )

    def _iter_batches(self, *, shuffle_batches: bool) -> Iterator:
        row_offset = 0
        for chunk_index, table in enumerate(self._iter_tables()):
            raw_rows = int(table.num_rows)
            state: dict[str, Any] = {
                "table": table,
                "raw_num_rows": raw_rows,
                "chunk_index": int(chunk_index),
            }
            state = dict(self._run_stage_sequence(state=state))
            self.diagnostics.record_chunk(raw_num_rows=raw_rows, state=state)

            chunk = state.get("chunk_out")
            if chunk is None:
                row_offset += raw_rows
                continue

            if "graph_event_id" in chunk and row_offset != 0:
                chunk["graph_event_id"] = chunk["graph_event_id"] + int(row_offset)

            num_graphs = int(chunk["num_graphs"])
            if num_graphs <= 0:
                row_offset += raw_rows
                continue

            starts = torch.arange(0, num_graphs, self.batch_size, dtype=torch.int64)
            if shuffle_batches and starts.numel() > 1:
                starts = starts[torch.randperm(starts.numel())]

            for g0 in starts.tolist():
                g1 = min(g0 + self.batch_size, num_graphs)
                batch = self._slice_chunk_batch(chunk, g0, g1)
                self.record_batch(batch)
                yield batch
            row_offset += raw_rows

    def record_batch(self, batch) -> None:
        self.diagnostics.record_batch(batch=batch)

    def get_diagnostics_summary(self) -> dict:
        return self.diagnostics.summary()

    @abstractmethod
    def _slice_chunk_batch(self, chunk: dict, g0: int, g1: int):
        raise NotImplementedError
