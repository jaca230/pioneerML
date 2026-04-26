from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping
from typing import Any

from pioneerml.data_loader.loaders import LoaderFactory


class BaseLoaderManager(ABC):
    def __init__(self, *, config: Mapping[str, Any] | None = None) -> None:
        self.config: dict[str, Any] = dict(config or {})

    @property
    def loader_factory(self) -> LoaderFactory:
        factory = self.config.get("loader_factory")
        if not isinstance(factory, LoaderFactory):
            raise RuntimeError("Loader manager requires config['loader_factory'] as LoaderFactory.")
        return factory

    @property
    def loaders(self) -> dict[str, Any]:
        raw = self.config.get("loaders")
        if raw is None:
            return {}
        if not isinstance(raw, Mapping):
            raise TypeError("Loader manager config['loaders'] must be a mapping when provided.")
        return dict(raw)

    @staticmethod
    def loader_key_for_purpose(*, purpose: str) -> str:
        return f"{str(purpose).strip().lower()}_loader"

    @abstractmethod
    def resolve_loader_params(
        self,
        *,
        purpose: str,
        forced_batch_size: int | None = None,
        default_batch_size: int = 64,
        default_chunk_row_groups: int = 4,
        default_mode: str = "train",
    ) -> dict[str, Any]:
        raise NotImplementedError

    def build_provider(
        self,
        *,
        purpose: str,
        forced_batch_size: int | None = None,
        default_batch_size: int = 64,
        default_chunk_row_groups: int = 4,
        default_mode: str = "train",
    ):
        params = self.resolve_loader_params(
            purpose=purpose,
            forced_batch_size=forced_batch_size,
            default_batch_size=default_batch_size,
            default_chunk_row_groups=default_chunk_row_groups,
            default_mode=default_mode,
        )
        provider = self.loader_factory.build(config=params)
        return provider, params

    def build_dataloader(
        self,
        *,
        purpose: str,
        forced_batch_size: int | None = None,
        default_batch_size: int = 64,
        default_chunk_row_groups: int = 4,
        default_mode: str = "train",
        default_shuffle: bool = False,
    ):
        provider, params = self.build_provider(
            purpose=purpose,
            forced_batch_size=forced_batch_size,
            default_batch_size=default_batch_size,
            default_chunk_row_groups=default_chunk_row_groups,
            default_mode=default_mode,
        )
        shuffle_batches = bool(params.get("shuffle_batches", default_shuffle))
        shuffle_within_batch_raw = params.get("shuffle_within_batch", None)
        shuffle_within_batch = None if shuffle_within_batch_raw is None else bool(shuffle_within_batch_raw)
        drop_remainders_raw = params.get("drop_remainders", params.get("drop_last", False))
        drop_remainders = bool(drop_remainders_raw)
        debug_epoch_batch_summary = bool(params.get("debug_epoch_batch_summary", False))
        worker_start_method_raw = params.get("worker_start_method", None)
        worker_start_method = None if worker_start_method_raw in (None, "", "none", "None") else str(worker_start_method_raw)
        persistent_workers_raw = params.get("persistent_workers", None)
        persistent_workers = None if persistent_workers_raw is None else bool(persistent_workers_raw)
        prefetch_factor_raw = params.get("prefetch_factor", None)
        prefetch_factor = None if prefetch_factor_raw is None else int(prefetch_factor_raw)
        torch_sharing_strategy_raw = params.get("torch_sharing_strategy", None)
        torch_sharing_strategy = (
            None
            if torch_sharing_strategy_raw in (None, "", "none", "None")
            else str(torch_sharing_strategy_raw)
        )
        loader = provider.make_dataloader(
            shuffle_batches=shuffle_batches,
            shuffle_within_batch=shuffle_within_batch,
            drop_remainders=drop_remainders,
            debug_epoch_batch_summary=debug_epoch_batch_summary,
            worker_start_method=worker_start_method,
            persistent_workers=persistent_workers,
            prefetch_factor=prefetch_factor,
            torch_sharing_strategy=torch_sharing_strategy,
        )
        return provider, params, loader
