from __future__ import annotations

import torch

from .base_model_handle import BaseModelHandle
from .registry import REGISTRY as MODEL_HANDLE_REGISTRY


@MODEL_HANDLE_REGISTRY.register("jit")
@MODEL_HANDLE_REGISTRY.register("torchscript")
@MODEL_HANDLE_REGISTRY.register("script")
class TorchScriptModelHandle(BaseModelHandle):
    TYPE = "script"

    def load(self, *, device: torch.device):
        scripted = torch.jit.load(str(self.path), map_location=device)
        scripted.eval()
        return scripted
