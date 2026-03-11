from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Callable

from .payloads import ExportStepPayload
from .utils.export_utils import call_export, json_safe, resolve_tensor_last_dim

from ..base_pipeline_step import BasePipelineStep


class BaseExportStep(BasePipelineStep):
    DEFAULT_CONFIG = {"enabled": True}

    def _execute(self) -> dict:
        raise NotImplementedError(f"{self.__class__.__name__} must implement _execute(...).")

    def _json_safe(self, value):
        return json_safe(value)

    @staticmethod
    def _resolve_tensor_last_dim(obj, *names: str) -> int:
        return resolve_tensor_last_dim(obj, *names)

    def export_torchscript(
        self,
        *,
        module,
        dataset,
        cfg: dict,
        pipeline_config: dict | None,
        hpo_params: dict | None,
        metrics: dict | None,
        default_export_dir: str,
        default_prefix: str,
        example_builder: Callable | None = None,
    ) -> dict:
        if cfg.get("enabled") is False:
            return self.build_payload(torchscript_path=None, metadata_path=None, skipped=True)

        export_dir = Path(cfg.get("export_dir", default_export_dir))
        export_dir.mkdir(parents=True, exist_ok=True)
        prefix = cfg.get("filename_prefix", default_prefix)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        torchscript_path = export_dir / f"{prefix}_{timestamp}_torchscript.pt"
        meta_path = export_dir / f"{prefix}_{timestamp}_meta.json"

        prefer_cuda = bool(cfg.get("prefer_cuda", True))
        export_fn = getattr(module.model, "export_torchscript", None)
        if export_fn is None:
            return self.build_payload(torchscript_path=None, metadata_path=None, skipped=True)

        self._call_export(
            export_fn=export_fn,
            torchscript_path=torchscript_path,
            prefer_cuda=prefer_cuda,
            cfg=cfg,
            dataset=dataset,
            example_builder=example_builder,
        )

        bundle_inputs = getattr(dataset, "inputs", None)
        if bundle_inputs is None:
            bundle_inputs = getattr(dataset, "data", None)
        bundle_targets = getattr(dataset, "targets", None)
        if bundle_targets is None and bundle_inputs is not None:
            bundle_targets = getattr(bundle_inputs, "y_graph", None)
            if bundle_targets is None:
                bundle_targets = getattr(bundle_inputs, "y_node", None)
            if bundle_targets is None:
                bundle_targets = getattr(bundle_inputs, "y", None)
        x_dim = self._resolve_tensor_last_dim(bundle_inputs, "x_node", "x")
        edge_attr_dim = self._resolve_tensor_last_dim(bundle_inputs, "x_edge", "edge_attr")
        num_classes = 0
        if bundle_targets is not None and hasattr(bundle_targets, "shape") and len(bundle_targets.shape) >= 2:
            num_classes = int(bundle_targets.shape[-1])

        meta = {
            "timestamp": timestamp,
            "torchscript_path": str(torchscript_path),
            "pipeline_config": self._json_safe(pipeline_config or {}),
            "export_config": self._json_safe(cfg or {}),
            "hpo_params": self._json_safe(hpo_params or {}),
            "metrics": self._json_safe(metrics or {}),
            "data_shapes": {
                "x_dim": int(x_dim),
                "edge_attr_dim": int(edge_attr_dim),
                "num_classes": int(num_classes),
            },
        }
        with meta_path.open("w", encoding="utf-8") as handle:
            json.dump(meta, handle, indent=2, sort_keys=True)

        return self.build_payload(
            torchscript_path=str(torchscript_path),
            metadata_path=str(meta_path),
        )

    def _call_export(
        self,
        *,
        export_fn,
        torchscript_path: Path,
        prefer_cuda: bool,
        cfg: dict,
        dataset,
        example_builder: Callable | None,
    ) -> None:
        call_export(
            export_fn=export_fn,
            torchscript_path=torchscript_path,
            prefer_cuda=prefer_cuda,
            cfg=cfg,
            dataset=dataset,
            example_builder=example_builder,
        )

    @staticmethod
    def build_payload(
        *,
        torchscript_path: str | None,
        metadata_path: str | None,
        **kwargs,
    ) -> ExportStepPayload:
        payload = {
            "torchscript_path": torchscript_path,
            "metadata_path": metadata_path,
        }
        payload.update(dict(kwargs))
        return ExportStepPayload(**payload)
