from __future__ import annotations

import torch

from .base_model_handle import BaseModelHandle
from .registry import REGISTRY as MODEL_HANDLE_REGISTRY


@MODEL_HANDLE_REGISTRY.register("traced")
@MODEL_HANDLE_REGISTRY.register("trace")
class TorchTraceModelHandle(BaseModelHandle):
    TYPE = "trace"

    def load(self, *, device: torch.device):
        traced = torch.jit.load(str(self.path), map_location=device)
        traced.eval()
        return traced
