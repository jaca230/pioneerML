from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterator
from typing import Any

import torch
from torch.utils.data import DataLoader, IterableDataset

from .config import DataFlowConfig, SplitSampleConfig
from .input_source import InputBackend, InputSourceSet, ParquetInputBackend, create_input_backend


class BaseLoader(ABC):
    """Abstract loader: input stream -> training/inference batch tensors."""

    MODE_TRAIN = "train"
    MODE_INFERENCE = "inference"

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
        _ = loader_params
        return cls(
            input_sources=input_sources,
            mode=mode,
            data_flow_config=data_flow_config,
            split_config=split_config,
            input_backend=create_input_backend(input_backend_name),
        )

    def __init__(
        self,
        *,
        input_sources: InputSourceSet,
        mode: str | None = None,
        data_flow_config: DataFlowConfig | None = None,
        split_config: SplitSampleConfig | None = None,
        input_backend: InputBackend | None = None,
    ) -> None:
        self.input_sources = input_sources
        self.input_backend = input_backend if input_backend is not None else ParquetInputBackend()

        self.data_flow_config = data_flow_config if data_flow_config is not None else DataFlowConfig()
        self.batch_size = int(self.data_flow_config.batch_size)
        self.num_workers = int(self.data_flow_config.num_workers)
        self.row_groups_per_chunk = int(self.data_flow_config.row_groups_per_chunk)

        self.split_config = split_config if split_config is not None else SplitSampleConfig()
        self.split = self.split_config.split
        self.train_fraction = float(self.split_config.train_fraction)
        self.val_fraction = float(self.split_config.val_fraction)
        self.test_fraction = float(self.split_config.test_fraction)
        self.split_seed = None if self.split_config.split_seed is None else int(self.split_config.split_seed)
        self.sample_fraction = self.split_config.sample_fraction

        mode_default = getattr(self, "mode", self.MODE_TRAIN) if mode is None else mode
        mode_norm = str(mode_default).strip().lower()
        if mode_norm not in {self.MODE_TRAIN, self.MODE_INFERENCE}:
            raise ValueError(
                f"Unsupported mode: {mode_default}. Expected '{self.MODE_TRAIN}' or '{self.MODE_INFERENCE}'."
            )
        self.mode = mode_norm

    @property
    def include_targets(self) -> bool:
        return str(self.mode).strip().lower() != self.MODE_INFERENCE

    def make_dataloader(self, *, shuffle_batches: bool) -> DataLoader:
        ds = _LoaderIterable(self, shuffle_batches=bool(shuffle_batches))
        kwargs: dict[str, object] = {
            "batch_size": None,
            "num_workers": int(self.num_workers),
            "pin_memory": True,
        }
        if int(self.num_workers) > 0:
            kwargs["persistent_workers"] = True
            kwargs["prefetch_factor"] = 2
        return DataLoader(ds, **kwargs)

    def build_inference_model_input(
        self,
        *,
        batch,
        device: torch.device,
        cfg: dict[str, Any],
    ) -> tuple[tuple[Any, ...], dict[str, Any]]:
        _ = batch
        _ = device
        _ = cfg
        raise NotImplementedError(
            f"{self.__class__.__name__} must implement build_inference_model_input(...) for inference usage."
        )

    def record_batch(self, batch) -> None:
        _ = batch

    def get_diagnostics_summary(self) -> dict:
        return {}

    @abstractmethod
    def _iter_batches(self, *, shuffle_batches: bool) -> Iterator:
        raise NotImplementedError


class _LoaderIterable(IterableDataset):
    def __init__(self, loader: BaseLoader, *, shuffle_batches: bool) -> None:
        self._loader = loader
        self._shuffle_batches = bool(shuffle_batches)

    def __iter__(self):
        yield from self._loader._iter_batches(shuffle_batches=self._shuffle_batches)
