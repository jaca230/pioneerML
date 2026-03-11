from __future__ import annotations

import json
import logging
import os
from typing import Any
from typing import Generic, TypeVar

from pioneerml.common.data_loader.config import DataFlowConfig, SplitSampleConfig
from pioneerml.common.data_loader.input_source import InputSourceSet
from .registry import resolve_loader

L = TypeVar("L")
LOGGER = logging.getLogger(__name__)


class LoaderFactory(Generic[L]):
    def __init__(
        self,
        *,
        loader_cls: type[L] | None = None,
        loader_name: str | None = None,
        input_sources: InputSourceSet,
        input_backend_name: str = "parquet",
        default_mode: str = "train",
    ) -> None:
        if loader_cls is None and loader_name is None:
            raise ValueError("LoaderFactory requires either loader_cls or loader_name.")
        self.loader_cls = loader_cls
        self.loader_name = None if loader_name is None else str(loader_name).strip()
        self.input_sources = input_sources
        self.input_backend_name = str(input_backend_name)
        self.default_mode = str(default_mode)

    @staticmethod
    def _as_optional_split(value) -> str | None:
        if value in (None, "", "none", "None"):
            return None
        return str(value).strip().lower()

    @classmethod
    def _default_chunk_workers(cls) -> int:
        _ = cls
        cpu = int(os.cpu_count() or 1)
        return max(1, cpu - 1)

    def _split_config_from_loader_params(self, *, loader_params: dict) -> SplitSampleConfig:
        cfg = dict(loader_params or {})
        split_seed_raw = cfg.get("split_seed", None)
        split_seed = None if split_seed_raw in (None, "", "none", "None") else int(split_seed_raw)
        sample_fraction_raw = cfg.get("sample_fraction")
        sample_fraction = None if sample_fraction_raw in (None, "", "none", "None") else float(sample_fraction_raw)
        return SplitSampleConfig(
            split=self._as_optional_split(cfg.get("split")),
            train_fraction=float(cfg.get("train_fraction", 0.9)),
            val_fraction=float(cfg.get("val_fraction", 0.05)),
            test_fraction=float(cfg.get("test_fraction", 0.05)),
            split_seed=split_seed,
            sample_fraction=sample_fraction,
        )

    def _data_flow_config_from_loader_params(self, *, loader_params: dict) -> DataFlowConfig:
        cfg = dict(loader_params or {})
        chunk_workers = cfg.get("chunk_workers", cfg.get("num_workers", None))
        if chunk_workers is None:
            num_workers = self._default_chunk_workers()
        else:
            num_workers = max(0, int(chunk_workers))
        return DataFlowConfig(
            batch_size=max(1, int(cfg.get("batch_size", 64))),
            row_groups_per_chunk=max(1, int(cfg.get("chunk_row_groups", cfg.get("row_groups_per_chunk", 4)))),
            num_workers=num_workers,
        )

    @classmethod
    def _resolve_loader_runtime(
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
            num_workers = cls._default_chunk_workers()
        else:
            num_workers = max(0, int(config_json.get("chunk_workers", config_json.get("num_workers", 0))))
        return mode, DataFlowConfig(
            batch_size=batch_size,
            row_groups_per_chunk=row_groups_per_chunk,
            num_workers=num_workers,
        )

    @classmethod
    def _resolve_loader_params(
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
            merged.setdefault("chunk_workers", cls._default_chunk_workers())
        else:
            merged.setdefault("chunk_workers", int(chunk_workers_cfg))
        merged.setdefault("mode", str(cfg.get("mode", default_mode)))
        return merged

    @classmethod
    def _ensure_loader_factory(cls, dataset: Any, *, expected_type: type | None = None):
        _ = cls
        factory = getattr(dataset, "loader_factory", None) or getattr(dataset, "loader", None)
        if expected_type is not None and not isinstance(factory, expected_type):
            raise RuntimeError(
                f"Dataset loader_factory has type {type(factory).__name__}; expected {expected_type.__name__}."
            )
        if factory is None:
            raise RuntimeError("Dataset is missing loader_factory/loader.")
        if not isinstance(factory, LoaderFactory):
            raise RuntimeError(
                f"Dataset loader_factory has type {type(factory).__name__}; expected LoaderFactory-compatible instance."
            )
        return factory

    @staticmethod
    def log_diagnostics(*, label: str, loader_provider: Any) -> dict:
        if loader_provider is None or not hasattr(loader_provider, "get_diagnostics_summary"):
            return {}
        try:
            summary = loader_provider.get_diagnostics_summary() or {}
        except Exception:
            return {}
        if summary:
            LOGGER.info("[loader_diagnostics][%s] %s", label, json.dumps(summary, sort_keys=True))
        return summary

    def _resolve_input_backend_name(self, *, loader_params: dict) -> str:
        cfg = dict(loader_params or {})
        return str(cfg.get("input_backend_name", self.input_backend_name))

    def _resolve_mode(self, *, loader_params: dict) -> str:
        cfg = dict(loader_params or {})
        return str(cfg.get("mode", self.default_mode))

    def _resolve_loader_class(self) -> type[L]:
        if self.loader_cls is not None:
            return self.loader_cls
        if self.loader_name is None:
            raise RuntimeError("LoaderFactory has neither loader_cls nor loader_name.")
        return resolve_loader(self.loader_name)

    def build_loader(self, *, loader_params: dict) -> L:
        split_cfg = self._split_config_from_loader_params(loader_params=loader_params)
        data_flow_cfg = self._data_flow_config_from_loader_params(loader_params=loader_params)
        loader_cls = self._resolve_loader_class()
        if not hasattr(loader_cls, "from_factory"):
            raise RuntimeError(
                f"{loader_cls.__name__} must implement classmethod from_factory(...) for LoaderFactory."
            )
        return loader_cls.from_factory(
            input_sources=self.input_sources,
            input_backend_name=self._resolve_input_backend_name(loader_params=loader_params),
            mode=self._resolve_mode(loader_params=loader_params),
            data_flow_config=data_flow_cfg,
            split_config=split_cfg,
            loader_params=dict(loader_params or {}),
        )
