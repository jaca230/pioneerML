from __future__ import annotations

import torch

from .base_model_handle import BaseModelHandle


@BaseModelHandle.register()
class TorchExportModelHandle(BaseModelHandle):
    TYPE = "export"
    ALIASES = ("torch_export", "torchexport")

    def load(self, *, device: torch.device):
        if not hasattr(torch, "export") or not hasattr(torch.export, "load"):
            raise RuntimeError("torch.export.load is unavailable in this torch build.")

        exported_program = torch.export.load(str(self.path))
        module = exported_program.module()
        module.eval()
        if hasattr(module, "to"):
            module = module.to(device)
        return module
