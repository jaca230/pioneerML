from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from ......resolver import BaseConfigResolver


class ExportRuntimeConfigResolver(BaseConfigResolver):
    def resolve(self, *, cfg: dict[str, Any]) -> None:
        cfg["export_type"] = self._resolve_export_type(raw=cfg.get("export_type", "script"))
        missing = [key for key in ("export_dir", "filename_prefix") if key not in cfg]
        if missing:
            raise KeyError(f"export missing required keys: {missing}")

        cfg["enabled"] = bool(cfg.get("enabled", True))
        cfg["prefer_cuda"] = bool(cfg.get("prefer_cuda", True))

        export_dir = cfg.get("export_dir")
        if export_dir is None or str(export_dir).strip() == "":
            raise ValueError("export.export_dir must be a non-empty string.")
        cfg["export_dir"] = str(export_dir)

        filename_prefix = cfg.get("filename_prefix")
        if filename_prefix is None or str(filename_prefix).strip() == "":
            raise ValueError("export.filename_prefix must be a non-empty string.")
        cfg["filename_prefix"] = str(filename_prefix)

        cfg["loader_config"] = self._resolve_loader_config(raw=cfg.get("loader_config"))

    @staticmethod
    def _resolve_loader_config(*, raw: Any) -> dict[str, Any]:
        if raw is None:
            return {
                "base": {
                    "batch_size": 1,
                    "chunk_row_groups": 1,
                    "chunk_workers": 0,
                },
                "export": {
                    "mode": "train",
                    "shuffle_batches": False,
                    "log_diagnostics": False,
                },
            }
        if not isinstance(raw, Mapping):
            raise TypeError("export.loader_config must be a mapping when provided.")

        base_cfg = dict(raw.get("base") or {})
        export_cfg = dict(raw.get("export") or {})
        if not isinstance(base_cfg, dict) or not isinstance(export_cfg, dict):
            raise TypeError("export.loader_config.base/export must be mappings.")

        out_base = {
            "batch_size": max(1, int(base_cfg.get("batch_size", 1))),
            "chunk_row_groups": max(1, int(base_cfg.get("chunk_row_groups", 1))),
            "chunk_workers": int(base_cfg.get("chunk_workers", 0)),
        }
        out_export = {
            "mode": str(export_cfg.get("mode", "train")),
            "shuffle_batches": bool(export_cfg.get("shuffle_batches", False)),
            "log_diagnostics": bool(export_cfg.get("log_diagnostics", False)),
        }
        return {"base": out_base, "export": out_export}

    @staticmethod
    def _resolve_export_type(*, raw: Any) -> str:
        value = str(raw).strip().lower()
        aliases = {
            "script": "script",
            "torchscript": "script",
            "trace": "trace",
            "tracing": "trace",
            "torch_export": "torch_export",
            "export": "torch_export",
            "pt2": "torch_export",
        }
        resolved = aliases.get(value)
        if resolved is None:
            allowed = "script, trace, torch_export"
            raise ValueError(f"export.export_type must be one of: [{allowed}]. Got: {raw!r}")
        return resolved
