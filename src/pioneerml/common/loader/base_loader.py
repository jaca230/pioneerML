from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterator

from torch.utils.data import DataLoader, IterableDataset


class BaseLoader(ABC):
    """Abstract loader: input stream -> training/inference batch tensors."""

    MODE_TRAIN = "train"
    MODE_INFERENCE = "inference"

    def __init__(self, *, batch_size: int = 64, num_workers: int = 0, mode: str | None = None) -> None:
        self.batch_size = max(1, int(batch_size))
        self.num_workers = max(0, int(num_workers))
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
