from __future__ import annotations

from abc import abstractmethod
from collections.abc import Iterator, MutableMapping
from typing import Any

import torch

from ..parquet_loader import ParquetLoader
from ...stage.stage_observers import CompositeStageObserver, JsonlObserver, MemoryObserver, StageObserver, TimingObserver
from ...stage.stage_runner import StageRunner
from ...stage.stages import BaseStage
from ...stage.utils.loader_diagnostics import LoaderDiagnostics


class StructuredLoader(ParquetLoader):
    """Parquet loader with a configurable stage pipeline."""

    def __init__(
        self,
        *args,
        stage_overrides: dict[str, BaseStage] | None = None,
        stage_observer: StageObserver | None = None,
        profiling: dict[str, Any] | None = None,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.stage_overrides = dict(stage_overrides or {})
        self.profiling = dict(profiling or {})
        self.diagnostics = LoaderDiagnostics(loader_kind=self.__class__.__name__)
        self.stage_sequence = self._build_stage_sequence()
        self.stage_runner = StageRunner(
            stages=self.stage_sequence,
            observer=self._build_stage_observer(stage_observer=stage_observer),
        )

    @abstractmethod
    def default_stage_order(self) -> list[str]:
        raise NotImplementedError

    @abstractmethod
    def default_stages(self) -> dict[str, BaseStage]:
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
        return self.stage_runner.run(
            state=state,
            loader=self,
            chunk_index=chunk_index,
            raw_num_rows=raw_num_rows,
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

            if "graph_event_ids" in chunk and row_offset != 0:
                chunk["graph_event_ids"] = chunk["graph_event_ids"] + int(row_offset)

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
