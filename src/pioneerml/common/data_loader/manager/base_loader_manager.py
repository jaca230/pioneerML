from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping
from typing import Any

from pioneerml.common.data_loader.loaders import LoaderFactory


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
        loader = provider.make_dataloader(shuffle_batches=bool(params.get("shuffle_batches", default_shuffle)))
        return provider, params, loader

