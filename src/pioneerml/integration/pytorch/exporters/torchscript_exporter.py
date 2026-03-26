from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch

from .base_exporter import BaseExporter
from .factory.registry import REGISTRY as EXPORTER_REGISTRY


@EXPORTER_REGISTRY.register("script")
@EXPORTER_REGISTRY.register("torchscript")
class TorchScriptExporter(BaseExporter):
    @property
    def export_type(self) -> str:
        return "script"

    @property
    def artifact_suffix(self) -> str:
        return "torchscript.pt"

    def export(
        self,
        *,
        model_obj: Any,
        output_path: Path,
        prefer_cuda: bool,
        cfg: Mapping[str, Any],
        dataset: Any,
        loader_provider: Any,
    ) -> None:
        _ = cfg, dataset
        if model_obj is None:
            raise RuntimeError("Cannot export: model object is missing.")
        example = self.build_example(loader_provider=loader_provider)
        custom_fn = self._resolve_custom_export_fn(
            model_obj=model_obj,
            function_names=("export_torchscript", "export_script"),
        )
        if callable(custom_fn):
            self._call_custom_export(
                export_fn=custom_fn,
                output_path=Path(output_path),
                example=example,
                prefer_cuda=bool(prefer_cuda),
            )
            return
        model = model_obj.eval()
        if hasattr(model, "to"):
            model = model.to("cpu")
        scripted = torch.jit.script(model)
        scripted.save(str(output_path))
