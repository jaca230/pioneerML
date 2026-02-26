from __future__ import annotations

from abc import ABC, abstractmethod


class BaseGraphLoaderFactory(ABC):
    def __init__(self, *, parquet_paths: list[str]) -> None:
        self.parquet_paths = [str(p) for p in parquet_paths]

    @abstractmethod
    def build_loader(self, *, loader_params: dict):
        raise NotImplementedError

