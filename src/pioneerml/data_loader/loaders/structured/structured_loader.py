from __future__ import annotations

from abc import abstractmethod
from collections.abc import Iterator, Mapping, MutableMapping
import inspect
import logging
from typing import Any

import torch

from pioneerml.data_loader.loaders.array_store.ndarray_store import NDArrayColumnSpec
from pioneerml.data_loader.loaders.array_store.schemas import LoaderSchema
from pioneerml.data_loader.loaders.base_loader import BaseLoader
from pioneerml.data_loader.loaders.config import DataFlowConfig, SplitSampleConfig
from pioneerml.data_loader.loaders.input_source import InputBackend, InputSourceSet
from pioneerml.data_loader.loaders.stage.loader_diagnostics import LoaderDiagnostics
from pioneerml.data_loader.loaders.stage.loader_stage_context import LoaderStageContext
from pioneerml.staged_runtime import PhaseRunner, StageRunner
from pioneerml.staged_runtime.stage_observers import (
    CompositeStageObserver,
    JsonlObserver,
    MemoryObserver,
    StageObserver,
    TimingObserver,
)
from pioneerml.data_loader.loaders.stage.stages import BaseStage

LOGGER = logging.getLogger(__name__)


class StructuredLoader(BaseLoader):
    """Structured staged loader using an input backend contract."""

    @staticmethod
    def _normalize_optional_nonnegative_int(value: object) -> int | None:
        if value in (None, "", "none", "None"):
            return None
        out = int(value)
        return 0 if out <= 0 else out

    @classmethod
    def _apply_common_loader_params(cls, *, loader, loader_params: Mapping[str, Any] | None):
        params = dict(loader_params or {})
        if "edge_template_cache_enabled" in params:
            setattr(loader, "edge_template_cache_enabled", bool(params.get("edge_template_cache_enabled")))
        if "edge_template_cache_max_entries" in params:
            setattr(
                loader,
                "edge_template_cache_max_entries",
                cls._normalize_optional_nonnegative_int(params.get("edge_template_cache_max_entries")),
            )
        if "debug_epoch_batch_summary" in params:
            setattr(loader, "debug_epoch_batch_summary_enabled", bool(params.get("debug_epoch_batch_summary")))
        return loader

    @classmethod
    def from_factory(
        cls,
        *,
        input_sources: InputSourceSet,
        input_backend_name: str,
        mode: str,
        data_flow_config: DataFlowConfig,
        split_config: SplitSampleConfig,
        loader_params: dict[str, Any] | None = None,
    ):
        params = dict(loader_params or {})
        stage_overrides = params.get("stage_overrides")
        stage_observer = params.get("stage_observer")
        profiling = dict(params.get("profiling") or {})
        ctor_kwargs: dict[str, Any] = {
            "input_sources": input_sources,
            "mode": mode,
            "data_flow_config": data_flow_config,
            "split_config": split_config,
            "input_backend": params.get("input_backend"),
            "input_backend_name": input_backend_name,
            "stage_overrides": stage_overrides if isinstance(stage_overrides, dict) else None,
            "stage_observer": stage_observer if isinstance(stage_observer, StageObserver) else None,
            "profiling": profiling,
        }
        allowed = set(inspect.signature(cls.__init__).parameters.keys())
        allowed.discard("self")
        filtered_kwargs = {k: v for k, v in ctor_kwargs.items() if k in allowed}
        loader = cls(**filtered_kwargs)
        return cls._apply_common_loader_params(loader=loader, loader_params=params)

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
        super().__init__(
            input_sources=input_sources,
            mode=mode,
            data_flow_config=data_flow_config,
            split_config=split_config,
            input_backend=input_backend,
        )
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

        self.edge_populate_graph_block = 512
        self.edge_template_cache_enabled = bool(
            getattr(
                self,
                "edge_template_cache_enabled",
                getattr(self, "EDGE_TEMPLATE_CACHE_ENABLED", False),
            )
        )
        self.edge_template_cache_max_entries = self._normalize_optional_nonnegative_int(
            getattr(
                self,
                "edge_template_cache_max_entries",
                getattr(self, "EDGE_TEMPLATE_CACHE_MAX_ENTRIES", None),
            )
        )

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

    @staticmethod
    def _build_segment_index(*, ptr: torch.Tensor, graph_perm: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        counts = (ptr[1:] - ptr[:-1]).to(dtype=torch.int64)
        perm_counts = counts[graph_perm]
        spans: list[torch.Tensor] = []
        for g_old in graph_perm.tolist():
            start = int(ptr[g_old].item())
            stop = int(ptr[g_old + 1].item())
            if stop > start:
                spans.append(torch.arange(start, stop, dtype=torch.int64, device=ptr.device))
        if spans:
            idx = torch.cat(spans, dim=0)
        else:
            idx = torch.empty((0,), dtype=torch.int64, device=ptr.device)
        return idx, perm_counts

    def _shuffle_chunk_graphs(self, *, chunk: Mapping[str, Any]) -> dict[str, Any]:
        num_graphs = int(chunk.get("num_graphs", 0))
        node_ptr = chunk.get("node_ptr")
        edge_ptr = chunk.get("edge_ptr")
        edge_index = chunk.get("edge_index")
        if num_graphs <= 1:
            return dict(chunk)
        if not isinstance(node_ptr, torch.Tensor) or not isinstance(edge_ptr, torch.Tensor):
            return dict(chunk)
        if not isinstance(edge_index, torch.Tensor) or edge_index.dim() != 2:
            return dict(chunk)

        graph_perm = torch.randperm(num_graphs, device=node_ptr.device)
        node_idx, perm_node_counts = self._build_segment_index(ptr=node_ptr, graph_perm=graph_perm)
        edge_idx, perm_edge_counts = self._build_segment_index(ptr=edge_ptr, graph_perm=graph_perm)

        out = dict(chunk)
        total_nodes = int(node_ptr[-1].item()) if int(node_ptr.numel()) > 0 else 0
        total_edges = int(edge_ptr[-1].item()) if int(edge_ptr.numel()) > 0 else 0

        new_node_ptr = torch.zeros((num_graphs + 1,), dtype=torch.int64, device=node_ptr.device)
        if int(perm_node_counts.numel()) > 0:
            new_node_ptr[1:] = torch.cumsum(perm_node_counts, dim=0)
        new_edge_ptr = torch.zeros((num_graphs + 1,), dtype=torch.int64, device=edge_ptr.device)
        if int(perm_edge_counts.numel()) > 0:
            new_edge_ptr[1:] = torch.cumsum(perm_edge_counts, dim=0)

        permuted_edge_spans: list[torch.Tensor] = []
        for g_new, g_old in enumerate(graph_perm.tolist()):
            n_old_start = int(node_ptr[g_old].item())
            e_old_start = int(edge_ptr[g_old].item())
            e_old_stop = int(edge_ptr[g_old + 1].item())
            if e_old_stop <= e_old_start:
                continue
            e_local = edge_index[:, e_old_start:e_old_stop].to(dtype=torch.int64) - int(n_old_start)
            n_new_start = int(new_node_ptr[g_new].item())
            permuted_edge_spans.append(e_local + int(n_new_start))
        if permuted_edge_spans:
            out["edge_index"] = torch.cat(permuted_edge_spans, dim=1)
        else:
            out["edge_index"] = torch.empty((2, 0), dtype=torch.int64, device=edge_index.device)

        handled = {
            "edge_index",
            "node_ptr",
            "edge_ptr",
            "graph_ptr",
            "num_graphs",
            "node_graph_id",
            "edge_graph_id",
            "slice_ptr",
            "graph_slice_ptr",
            "slice_graph_id",
            "node_slice_id",
            "atar_slice_ptr",
            "atar_slice_pdg_target",
            "atar_slice_multi_target",
            "atar_slice_trigger_target",
            "atar_slice_start_target",
            "atar_slice_stop_target",
            "atar_angle_target",
            "atar_pion_stop_target",
            "atar_pion_stop_valid_target",
        }

        for key, value in list(out.items()):
            if key in handled:
                continue
            if not isinstance(value, torch.Tensor) or value.dim() == 0:
                continue
            if int(value.shape[0]) == total_nodes:
                out[key] = value.index_select(0, node_idx)
                continue
            if int(value.shape[0]) == total_edges:
                out[key] = value.index_select(0, edge_idx)
                continue
            if int(value.shape[0]) == num_graphs:
                out[key] = value.index_select(0, graph_perm)
                continue

        out["node_ptr"] = new_node_ptr
        out["edge_ptr"] = new_edge_ptr
        out["graph_ptr"] = torch.tensor([0, num_graphs], dtype=torch.int64, device=node_ptr.device)
        out["num_graphs"] = int(num_graphs)
        out["node_graph_id"] = torch.repeat_interleave(
            torch.arange(num_graphs, dtype=torch.int64, device=node_ptr.device),
            perm_node_counts,
        )
        if int(out["edge_index"].numel()) > 0:
            out["edge_graph_id"] = out["node_graph_id"][out["edge_index"][0]]
        else:
            out["edge_graph_id"] = torch.empty((0,), dtype=torch.int64, device=node_ptr.device)

        graph_slice_ptr = chunk.get("graph_slice_ptr")
        if isinstance(graph_slice_ptr, torch.Tensor) and int(graph_slice_ptr.numel()) == int(num_graphs + 1):
            slice_counts = (graph_slice_ptr[1:] - graph_slice_ptr[:-1]).to(dtype=torch.int64)
            perm_slice_counts = slice_counts[graph_perm]
            new_graph_slice_ptr = torch.zeros((num_graphs + 1,), dtype=torch.int64, device=graph_slice_ptr.device)
            if int(perm_slice_counts.numel()) > 0:
                new_graph_slice_ptr[1:] = torch.cumsum(perm_slice_counts, dim=0)
            out["graph_slice_ptr"] = new_graph_slice_ptr

            total_slices = int(graph_slice_ptr[-1].item()) if int(graph_slice_ptr.numel()) > 0 else 0
            slice_spans: list[torch.Tensor] = []
            for g_old in graph_perm.tolist():
                s0 = int(graph_slice_ptr[g_old].item())
                s1 = int(graph_slice_ptr[g_old + 1].item())
                if s1 > s0:
                    slice_spans.append(torch.arange(s0, s1, dtype=torch.int64, device=graph_slice_ptr.device))
            if slice_spans:
                slice_perm = torch.cat(slice_spans, dim=0)
            else:
                slice_perm = torch.empty((0,), dtype=torch.int64, device=graph_slice_ptr.device)

            old_to_new_slice = torch.full((max(1, total_slices),), -1, dtype=torch.int64, device=graph_slice_ptr.device)
            if int(slice_perm.numel()) > 0:
                old_to_new_slice[slice_perm] = torch.arange(int(slice_perm.numel()), dtype=torch.int64, device=graph_slice_ptr.device)

            for key, value in list(out.items()):
                if key in handled:
                    continue
                if not isinstance(value, torch.Tensor) or value.dim() == 0:
                    continue
                if int(value.shape[0]) == total_slices:
                    out[key] = value.index_select(0, slice_perm)

            out["slice_graph_id"] = torch.repeat_interleave(
                torch.arange(num_graphs, dtype=torch.int64, device=graph_slice_ptr.device),
                perm_slice_counts,
            )
            if "node_slice_id" in out and isinstance(out["node_slice_id"], torch.Tensor):
                old_node_slice = out["node_slice_id"]
                if int(old_node_slice.numel()) > 0 and int(slice_perm.numel()) > 0:
                    mapped = old_to_new_slice[old_node_slice.to(dtype=torch.int64)]
                    out["node_slice_id"] = mapped
            slice_ptr = chunk.get("slice_ptr")
            if isinstance(slice_ptr, torch.Tensor) and int(slice_ptr.numel()) == int(total_slices + 1):
                slice_node_counts = (slice_ptr[1:] - slice_ptr[:-1]).to(dtype=torch.int64)
                perm_slice_node_counts = slice_node_counts[slice_perm] if int(slice_perm.numel()) > 0 else torch.empty((0,), dtype=torch.int64, device=slice_ptr.device)
                new_slice_ptr = torch.zeros((int(perm_slice_node_counts.numel()) + 1,), dtype=torch.int64, device=slice_ptr.device)
                if int(perm_slice_node_counts.numel()) > 0:
                    new_slice_ptr[1:] = torch.cumsum(perm_slice_node_counts, dim=0)
                out["slice_ptr"] = new_slice_ptr

        atar_slice_ptr = chunk.get("atar_slice_ptr")
        if isinstance(atar_slice_ptr, torch.Tensor) and int(atar_slice_ptr.numel()) == int(num_graphs + 1):
            atar_slice_counts = (atar_slice_ptr[1:] - atar_slice_ptr[:-1]).to(dtype=torch.int64)
            perm_atar_counts = atar_slice_counts[graph_perm]
            new_atar_slice_ptr = torch.zeros((num_graphs + 1,), dtype=torch.int64, device=atar_slice_ptr.device)
            if int(perm_atar_counts.numel()) > 0:
                new_atar_slice_ptr[1:] = torch.cumsum(perm_atar_counts, dim=0)
            out["atar_slice_ptr"] = new_atar_slice_ptr

            total_atar = int(atar_slice_ptr[-1].item()) if int(atar_slice_ptr.numel()) > 0 else 0
            atar_spans: list[torch.Tensor] = []
            for g_old in graph_perm.tolist():
                a0 = int(atar_slice_ptr[g_old].item())
                a1 = int(atar_slice_ptr[g_old + 1].item())
                if a1 > a0:
                    atar_spans.append(torch.arange(a0, a1, dtype=torch.int64, device=atar_slice_ptr.device))
            if atar_spans:
                atar_perm = torch.cat(atar_spans, dim=0)
            else:
                atar_perm = torch.empty((0,), dtype=torch.int64, device=atar_slice_ptr.device)

            atar_keys = {
                "atar_slice_pdg_target",
                "atar_slice_multi_target",
                "atar_slice_trigger_target",
                "atar_slice_start_target",
                "atar_slice_stop_target",
                "atar_angle_target",
                "atar_pion_stop_target",
                "atar_pion_stop_valid_target",
            }
            for key in atar_keys:
                value = out.get(key)
                if isinstance(value, torch.Tensor) and value.dim() > 0 and int(value.shape[0]) == total_atar:
                    out[key] = value.index_select(0, atar_perm)

        return out

    def _iter_batches(
        self,
        *,
        shuffle_batches: bool,
        shuffle_within_batch: bool,
        drop_remainders: bool,
        debug_epoch_batch_summary: bool,
    ) -> Iterator:
        row_offset = 0
        epoch_batch_sizes: list[int] = []
        epoch_usable_graphs = 0
        epoch_dropped_remainder_count = 0
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

            if shuffle_within_batch and num_graphs > 1:
                chunk = self._shuffle_chunk_graphs(chunk=chunk)

            starts = torch.arange(0, num_graphs, self.batch_size, dtype=torch.int64)
            if shuffle_batches and starts.numel() > 1:
                starts = starts[torch.randperm(starts.numel())]

            for g0 in starts.tolist():
                g1 = min(g0 + self.batch_size, num_graphs)
                # Drop only true tail remainders from chunks that produced at least one full batch.
                # If a chunk has fewer than batch_size graphs total, keep it to avoid empty training.
                if (
                    drop_remainders
                    and num_graphs >= int(self.batch_size)
                    and (g1 - g0) < int(self.batch_size)
                ):
                    epoch_dropped_remainder_count += 1
                    continue
                batch_graphs = int(g1 - g0)
                epoch_batch_sizes.append(batch_graphs)
                epoch_usable_graphs += batch_graphs
                batch = self._slice_chunk_batch(chunk, g0, g1)
                self.record_batch(batch)
                yield batch
            row_offset += raw_rows
        if bool(debug_epoch_batch_summary):
            LOGGER.info(
                "loader_epoch_summary mode=%s split=%s usable_graphs=%d batch_sizes=%s dropped_remainder_count=%d",
                str(getattr(self, "mode", "")),
                str(getattr(self, "split", "")),
                int(epoch_usable_graphs),
                list(epoch_batch_sizes),
                int(epoch_dropped_remainder_count),
            )

    def record_batch(self, batch) -> None:
        self.diagnostics.record_batch(batch=batch)

    def get_diagnostics_summary(self) -> dict:
        return self.diagnostics.summary()

    @abstractmethod
    def _slice_chunk_batch(self, chunk: dict, g0: int, g1: int):
        raise NotImplementedError
