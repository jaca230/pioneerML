from __future__ import annotations

import torch

from .base_model_handle import BaseModelHandle
from .registry import REGISTRY as MODEL_HANDLE_REGISTRY


@MODEL_HANDLE_REGISTRY.register("torchexport")
@MODEL_HANDLE_REGISTRY.register("torch_export")
@MODEL_HANDLE_REGISTRY.register("export")
class TorchExportModelHandle(BaseModelHandle):
    TYPE = "export"

    def load(self, *, device: torch.device):
        if not hasattr(torch, "export") or not hasattr(torch.export, "load"):
            raise RuntimeError("torch.export.load is unavailable in this torch build.")

        exported_program = torch.export.load(str(self.path))
        module = exported_program.module()
        module.eval()
        if hasattr(module, "to"):
            module = module.to(device)
        return module
