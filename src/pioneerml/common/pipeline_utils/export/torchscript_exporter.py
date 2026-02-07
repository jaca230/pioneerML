from __future__ import annotations

import inspect
import json
from datetime import datetime
from pathlib import Path
from typing import Callable


class TorchscriptExporter:
    """Shared TorchScript export flow for training pipelines."""

    def export(
        self,
        *,
        module,
        dataset,
        cfg: dict,
        hpo_params: dict | None,
        metrics: dict | None,
        default_export_dir: str,
        default_prefix: str,
        example_builder: Callable | None = None,
    ) -> dict:
        if cfg.get("enabled") is False:
            return {"torchscript_path": None, "metadata_path": None, "skipped": True}

        export_dir = Path(cfg.get("export_dir", default_export_dir))
        export_dir.mkdir(parents=True, exist_ok=True)
        prefix = cfg.get("filename_prefix", default_prefix)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        torchscript_path = export_dir / f"{prefix}_{timestamp}_torchscript.pt"
        meta_path = export_dir / f"{prefix}_{timestamp}_meta.json"

        prefer_cuda = bool(cfg.get("prefer_cuda", True))
        export_fn = getattr(module.model, "export_torchscript", None)
        if export_fn is None:
            return {"torchscript_path": None, "metadata_path": None, "skipped": True}

        try:
            self._call_export(
                export_fn=export_fn,
                torchscript_path=torchscript_path,
                prefer_cuda=prefer_cuda,
                cfg=cfg,
                dataset=dataset,
                example_builder=example_builder,
            )
        except (TypeError, NotImplementedError):
            return {"torchscript_path": None, "metadata_path": None, "skipped": True}

        meta = {
            "timestamp": timestamp,
            "torchscript_path": str(torchscript_path),
            "hpo_params": hpo_params or {},
            "metrics": metrics or {},
            "data_shapes": {
                "x_dim": int(dataset.data.x.shape[-1]),
                "edge_attr_dim": int(dataset.data.edge_attr.shape[-1]),
                "num_classes": int(dataset.targets.shape[-1]),
            },
        }
        with meta_path.open("w", encoding="utf-8") as handle:
            json.dump(meta, handle, indent=2, sort_keys=True)

        return {
            "torchscript_path": str(torchscript_path),
            "metadata_path": str(meta_path),
        }

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
        sig = inspect.signature(export_fn)
        example = cfg.get("example")
        if example is None and example_builder is not None and "example" in sig.parameters:
            example = example_builder(dataset)

        if "example" in sig.parameters:
            if "prefer_cuda" in sig.parameters:
                export_fn(torchscript_path, example, prefer_cuda=prefer_cuda)
            else:
                export_fn(torchscript_path, example)
            return

        if "prefer_cuda" in sig.parameters:
            export_fn(torchscript_path, prefer_cuda=prefer_cuda)
            return

        export_fn(torchscript_path)
