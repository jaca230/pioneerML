from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch

from .base_exporter import BaseExporter
from .factory.registry import REGISTRY as EXPORTER_REGISTRY


@EXPORTER_REGISTRY.register("torch_export")
@EXPORTER_REGISTRY.register("export")
@EXPORTER_REGISTRY.register("pt2")
@EXPORTER_REGISTRY.register("torchexport")
class TorchExportProgramExporter(BaseExporter):
    @property
    def export_type(self) -> str:
        return "torch_export"

    @property
    def artifact_suffix(self) -> str:
        return "export.pt2"

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
            function_names=("export_torch_export", "export_exported_program"),
        )
        if callable(custom_fn):
            self._call_custom_export(
                export_fn=custom_fn,
                output_path=Path(output_path),
                example=example,
                prefer_cuda=bool(prefer_cuda),
            )
            return
        if not hasattr(torch, "export") or not hasattr(torch.export, "export") or not hasattr(torch.export, "save"):
            raise RuntimeError("torch.export is unavailable in this runtime.")
        model = model_obj.eval()
        if hasattr(model, "to"):
            model = model.to("cpu")
        args, kwargs = self._normalize_example_for_inputs(example=self._to_cpu(example))
        program = torch.export.export(model, args=args, kwargs=kwargs)
        torch.export.save(program, str(output_path))
