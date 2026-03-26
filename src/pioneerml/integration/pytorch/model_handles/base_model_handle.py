from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

import torch


class BaseModelHandle(ABC):
    """Serializable model-handle contract for deferred model loading."""

    TYPE: str = "base"

    def __init__(self, *, model_path: str) -> None:
        self.model_path = str(model_path)

    @property
    def path(self) -> Path:
        return Path(self.model_path).expanduser().resolve()

    @abstractmethod
    def load(self, *, device: torch.device):
        """Materialize and return an inference-ready model object."""
        raise NotImplementedError

    def to_payload(self) -> dict[str, str]:
        return {"type": str(self.TYPE), "model_path": str(self.path)}
