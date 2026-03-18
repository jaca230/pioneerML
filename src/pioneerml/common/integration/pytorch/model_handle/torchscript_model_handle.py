from __future__ import annotations

import torch

from .base_model_handle import BaseModelHandle


@BaseModelHandle.register()
class TorchScriptModelHandle(BaseModelHandle):
    TYPE = "script"
    ALIASES = ("torchscript", "jit")

    def load(self, *, device: torch.device):
        scripted = torch.jit.load(str(self.path), map_location=device)
        scripted.eval()
        return scripted
