from __future__ import annotations

import torch

from .base_model_handle import BaseModelHandle


@BaseModelHandle.register()
class TorchTraceModelHandle(BaseModelHandle):
    TYPE = "trace"
    ALIASES = ("traced",)

    def load(self, *, device: torch.device):
        traced = torch.jit.load(str(self.path), map_location=device)
        traced.eval()
        return traced
