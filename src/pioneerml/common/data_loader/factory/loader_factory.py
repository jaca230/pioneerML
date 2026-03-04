from __future__ import annotations

from typing import Generic, TypeVar

from pioneerml.common.data_loader.config import DataFlowConfig, SplitSampleConfig
from pioneerml.common.data_loader.input_source import InputSourceSet
from .registry import resolve_loader

L = TypeVar("L")


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
    def split_config_from_loader_params(cls, *, loader_params: dict) -> SplitSampleConfig:
        cfg = dict(loader_params or {})
        split_seed_raw = cfg.get("split_seed", None)
        split_seed = None if split_seed_raw in (None, "", "none", "None") else int(split_seed_raw)
        sample_fraction_raw = cfg.get("sample_fraction")
        sample_fraction = None if sample_fraction_raw in (None, "", "none", "None") else float(sample_fraction_raw)
        return SplitSampleConfig(
            split=cls._as_optional_split(cfg.get("split")),
            train_fraction=float(cfg.get("train_fraction", 0.9)),
            val_fraction=float(cfg.get("val_fraction", 0.05)),
            test_fraction=float(cfg.get("test_fraction", 0.05)),
            split_seed=split_seed,
            sample_fraction=sample_fraction,
        )

    @staticmethod
    def data_flow_config_from_loader_params(*, loader_params: dict) -> DataFlowConfig:
        cfg = dict(loader_params or {})
        return DataFlowConfig(
            batch_size=max(1, int(cfg.get("batch_size", 64))),
            row_groups_per_chunk=max(1, int(cfg.get("chunk_row_groups", cfg.get("row_groups_per_chunk", 4)))),
            num_workers=max(0, int(cfg.get("chunk_workers", cfg.get("num_workers", 0)))),
        )

    def resolve_input_backend_name(self, *, loader_params: dict) -> str:
        cfg = dict(loader_params or {})
        return str(cfg.get("input_backend_name", self.input_backend_name))

    @staticmethod
    def resolve_mode(*, loader_params: dict, default_mode: str = "train") -> str:
        cfg = dict(loader_params or {})
        return str(cfg.get("mode", default_mode))

    @staticmethod
    def resolve_profiling(*, loader_params: dict) -> dict:
        cfg = dict(loader_params or {})
        return dict(cfg.get("profiling") or {})

    def _resolve_loader_class(self) -> type[L]:
        if self.loader_cls is not None:
            return self.loader_cls
        if self.loader_name is None:
            raise RuntimeError("LoaderFactory has neither loader_cls nor loader_name.")
        return resolve_loader(self.loader_name)

    def build_loader(self, *, loader_params: dict) -> L:
        split_cfg = self.split_config_from_loader_params(loader_params=loader_params)
        data_flow_cfg = self.data_flow_config_from_loader_params(loader_params=loader_params)
        loader_cls = self._resolve_loader_class()
        return loader_cls(
            input_sources=self.input_sources,
            input_backend_name=self.resolve_input_backend_name(loader_params=loader_params),
            mode=self.resolve_mode(loader_params=loader_params, default_mode=self.default_mode),
            data_flow_config=data_flow_cfg,
            split_config=split_cfg,
            profiling=self.resolve_profiling(loader_params=loader_params),
        )
