from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq

from pioneerml.common.loader.config import DataFlowConfig
from pioneerml.common.parquet import ParquetInputSet

from ..base_pipeline_step import BasePipelineStep

LOGGER = logging.getLogger(__name__)


class BaseLoaderStep(BasePipelineStep):
    def default_config(self) -> dict:
        return {}

    @staticmethod
    def default_chunk_workers() -> int:
        cpu = int(os.cpu_count() or 1)
        return max(1, cpu - 1)

    @staticmethod
    def resolve_parquet_paths(parquet_paths: list[str]) -> list[str]:
        resolved = [str(Path(p).expanduser().resolve()) for p in parquet_paths]
        if not resolved:
            raise RuntimeError("No parquet paths provided.")
        return resolved

    @classmethod
    def resolve_parquet_input_set(cls, parquet_input_set: dict | ParquetInputSet) -> ParquetInputSet:
        if isinstance(parquet_input_set, ParquetInputSet):
            return parquet_input_set
        payload = dict(parquet_input_set or {})
        main_paths = payload.get("main_paths")
        optional_paths_by_name = payload.get("optional_paths_by_name")
        if not isinstance(main_paths, list):
            raise RuntimeError("parquet_input_set must include a list field 'main_paths'.")
        if optional_paths_by_name is not None and not isinstance(optional_paths_by_name, dict):
            raise RuntimeError("parquet_input_set.optional_paths_by_name must be a dict when provided.")
        return ParquetInputSet(
            main_paths=[str(p) for p in main_paths],
            optional_paths_by_name=(
                {str(k): v for k, v in optional_paths_by_name.items()} if optional_paths_by_name is not None else None
            ),
        )

    @staticmethod
    def count_parquet_rows(parquet_paths: list[str]) -> int:
        total = 0
        for p in parquet_paths:
            total += int(pq.ParquetFile(p).metadata.num_rows)
        return total

    @classmethod
    def resolve_parquet_runtime(
        cls,
        config_json: dict,
        *,
        default_mode: str = "inference",
        allowed_modes: tuple[str, ...] = ("inference", "train"),
        default_batch_size: int = 64,
        default_chunk_row_groups: int = 4,
    ) -> tuple[str, DataFlowConfig]:
        mode = str(config_json.get("mode", default_mode)).strip().lower()
        if mode not in set(allowed_modes):
            raise ValueError(f"Unsupported loader mode: {mode}. Expected one of {allowed_modes}.")
        batch_size = max(1, int(config_json.get("batch_size", default_batch_size)))
        row_groups_per_chunk = max(
            1, int(config_json.get("chunk_row_groups", config_json.get("row_groups_per_chunk", default_chunk_row_groups)))
        )
        if config_json.get("chunk_workers", None) is None and config_json.get("num_workers", None) is None:
            num_workers = cls.default_chunk_workers()
        else:
            num_workers = max(0, int(config_json.get("chunk_workers", config_json.get("num_workers", 0))))
        return mode, DataFlowConfig(
            batch_size=batch_size,
            row_groups_per_chunk=row_groups_per_chunk,
            num_workers=num_workers,
        )

    @staticmethod
    def ensure_loader_factory(dataset: Any, *, expected_type: type | None = None):
        factory = getattr(dataset, "loader_factory", None) or getattr(dataset, "loader", None)
        if expected_type is not None and not isinstance(factory, expected_type):
            raise RuntimeError(f"Dataset loader_factory has type {type(factory).__name__}; expected {expected_type.__name__}.")
        if factory is None:
            raise RuntimeError("Dataset is missing loader_factory/loader.")
        return factory

    @classmethod
    def resolve_loader_params(
        cls,
        cfg: dict,
        *,
        purpose: str,
        forced_batch_size: int | None = None,
        default_batch_size: int = 64,
        default_chunk_row_groups: int = 4,
        default_mode: str = "train",
    ) -> dict:
        raw = cfg.get("loader_config")
        base_cfg: dict = {}
        purpose_cfg: dict = {}
        if isinstance(raw, dict):
            if any(k in raw for k in ("base", "train", "val", "evaluate", "export", "inference")):
                base_cfg = dict(raw.get("base") or {})
                purpose_cfg = dict(raw.get(purpose) or {})
            else:
                base_cfg = dict(raw)
        merged = {**base_cfg, **purpose_cfg}

        if forced_batch_size is not None:
            merged["batch_size"] = int(forced_batch_size)
        else:
            merged.setdefault("batch_size", int(cfg.get("batch_size", default_batch_size)))
        merged.setdefault("chunk_row_groups", int(cfg.get("chunk_row_groups", default_chunk_row_groups)))
        chunk_workers_cfg = cfg.get("chunk_workers")
        if chunk_workers_cfg is None:
            merged.setdefault("chunk_workers", cls.default_chunk_workers())
        else:
            merged.setdefault("chunk_workers", int(chunk_workers_cfg))
        merged.setdefault("mode", str(cfg.get("mode", default_mode)))
        return merged

    @staticmethod
    def log_loader_diagnostics(*, label: str, loader_provider: Any) -> dict:
        if loader_provider is None or not hasattr(loader_provider, "get_diagnostics_summary"):
            return {}
        try:
            summary = loader_provider.get_diagnostics_summary() or {}
        except Exception:
            return {}
        if summary:
            LOGGER.info("[loader_diagnostics][%s] %s", label, json.dumps(summary, sort_keys=True))
        return summary
