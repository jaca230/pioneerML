from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from ......resolver import BaseConfigResolver


class ExportConfigResolver(BaseConfigResolver):
    def resolve(self, *, cfg: dict[str, Any]) -> None:
        exporter = self._resolve_exporter(raw=cfg.get("exporter"))
        exporter_cfg = dict(exporter.get("config") or {})

        exporter_cfg["enabled"] = bool(exporter_cfg.get("enabled", True))
        exporter_cfg["prefer_cuda"] = bool(exporter_cfg.get("prefer_cuda", True))

        export_dir = exporter_cfg.get("export_dir")
        if export_dir is None or str(export_dir).strip() == "":
            raise ValueError("export.exporter.config.export_dir must be a non-empty string.")
        exporter_cfg["export_dir"] = str(export_dir)

        filename_prefix = exporter_cfg.get("filename_prefix")
        if filename_prefix is None or str(filename_prefix).strip() == "":
            raise ValueError("export.exporter.config.filename_prefix must be a non-empty string.")
        exporter_cfg["filename_prefix"] = str(filename_prefix)

        exporter["config"] = exporter_cfg
        cfg["exporter"] = exporter

    @staticmethod
    def _resolve_exporter(*, raw: Any) -> dict[str, Any]:
        if not isinstance(raw, Mapping):
            raise TypeError("export.exporter must be a mapping with keys ['type', 'config'].")
        block = dict(raw)
        plugin_type = block.get("type")
        if not isinstance(plugin_type, str) or plugin_type.strip() == "":
            raise ValueError("export.exporter.type must be a non-empty string.")
        plugin_cfg = block.get("config")
        if plugin_cfg is None:
            plugin_cfg = {}
        if not isinstance(plugin_cfg, Mapping):
            raise TypeError("export.exporter.config must be a mapping.")
        return {"type": str(plugin_type).strip(), "config": dict(plugin_cfg)}
