from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import MutableMapping
from dataclasses import dataclass
from typing import Any

from ..config import WriterRunConfig
from ..base_data_writer import BaseDataWriter
from ..array_store import OutputSchema
from ..stage import WriterDiagnostics, WriterStageRunner
from ..stage.stages import BaseWriterStage
from pioneerml.common.staged_runtime.stage_observers import (
    CompositeStageObserver,
    JsonlObserver,
    MemoryObserver,
    StageObserver,
    TimingObserver,
)


@dataclass(frozen=True)
class WriterPhaseOrder:
    start: list[str]
    chunk: list[str]
    finalize: list[str]


@dataclass(frozen=True)
class WriterPhaseStages:
    start: dict[str, BaseWriterStage]
    chunk: dict[str, BaseWriterStage]
    finalize: dict[str, BaseWriterStage]


class StructuredDataWriter(BaseDataWriter, ABC):
    """Writer base for structured outputs with staged-run configuration."""

    @classmethod
    def from_factory(
        cls,
        *,
        output_backend_name: str,
        run_config: WriterRunConfig | None = None,
        writer_params: dict[str, Any] | None = None,
    ):
        if run_config is None:
            raise RuntimeError(f"{cls.__name__}.from_factory requires run_config.")
        params = dict(writer_params or {})
        return cls(
            output_backend=params.get("output_backend"),
            output_backend_name=output_backend_name,
            run_config=run_config,
            stage_overrides=params.get("stage_overrides"),
            stage_observer=params.get("stage_observer"),
            profiling=dict(params.get("profiling") or {}),
        )

    def __init__(
        self,
        *args,
        run_config: WriterRunConfig,
        stage_overrides: dict[str, BaseWriterStage] | None = None,
        stage_observer: StageObserver | None = None,
        profiling: dict[str, Any] | None = None,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.run_config = run_config
        self.stage_overrides = dict(stage_overrides or {})
        self.profiling = dict(profiling or {})
        self.diagnostics = WriterDiagnostics(writer_kind=self.__class__.__name__)
        self._stage_observer = self._build_stage_observer(stage_observer=stage_observer)
        self._stage_runner = self._build_stage_runner()
        self._run_state: dict[str, Any] = {}

    @abstractmethod
    def default_stage_order(self) -> WriterPhaseOrder:
        raise NotImplementedError

    @abstractmethod
    def default_stages(self) -> WriterPhaseStages:
        raise NotImplementedError

    @abstractmethod
    def output_schema(self) -> OutputSchema:
        raise NotImplementedError

    def _phase_stage_spec(self, phase: str) -> tuple[list[str], dict[str, BaseWriterStage]]:
        key = str(phase).strip().lower()
        order = self.default_stage_order()
        stages = self.default_stages()
        if key == "start":
            return list(order.start), dict(stages.start)
        if key == "chunk":
            return list(order.chunk), dict(stages.chunk)
        if key == "finalize":
            return list(order.finalize), dict(stages.finalize)
        raise ValueError("Unsupported writer phase. Allowed phases: ['start', 'chunk', 'finalize'].")

    def _build_stage_sequence(self, *, phase: str) -> list[BaseWriterStage]:
        order, default_stages = self._phase_stage_spec(phase)
        stages = dict(default_stages)
        stages.update(self.stage_overrides)
        missing = [name for name in order if name not in stages]
        if missing:
            raise RuntimeError(f"Missing writer stage implementations for: {missing}")
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

    def _build_stage_runner(self) -> WriterStageRunner:
        return WriterStageRunner(
            init_stages=self._build_stage_sequence(phase="start"),
            chunk_stages=self._build_stage_sequence(phase="chunk"),
            finalize_stages=self._build_stage_sequence(phase="finalize"),
            observer=self._stage_observer,
        )

    def _run_stage_sequence(
        self,
        *,
        state: MutableMapping[str, Any],
        phase: str,
    ) -> MutableMapping[str, Any]:
        chunk_index = int(state.get("chunk_index", 0))
        raw_num_rows = int(state.get("raw_num_rows", state.get("num_rows", 0)))
        state["chunk_index"] = chunk_index
        state["raw_num_rows"] = raw_num_rows
        phase_key = str(phase).strip().lower()
        if phase_key not in {"start", "chunk", "finalize"}:
            raise ValueError(f"Unsupported writer phase: {phase}")
        return self._stage_runner.run_phase(phase=phase_key, state=state, owner=self)

    def on_start(self, *, state: MutableMapping[str, Any]) -> MutableMapping[str, Any]:
        self._run_state = dict(state)
        out = self._run_stage_sequence(state=self._run_state, phase="start")
        self._run_state = dict(out)
        return out

    def on_chunk(self, *, state: MutableMapping[str, Any]) -> MutableMapping[str, Any]:
        merged = dict(self._run_state)
        merged.update(state)
        out = self._run_stage_sequence(state=merged, phase="chunk")
        self._run_state = dict(out)
        return out

    def on_finalize(self, *, state: MutableMapping[str, Any] | None = None) -> MutableMapping[str, Any]:
        merged = dict(self._run_state)
        if state is not None:
            merged.update(state)
        out = self._run_stage_sequence(state=merged, phase="finalize")
        self._run_state = dict(out)
        return out
