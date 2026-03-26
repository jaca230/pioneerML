from __future__ import annotations

from typing import Any

from .....resolver import BaseConfigResolver


class ModelHandleBuilderConfigResolver(BaseConfigResolver):
    def resolve(self, *, cfg: dict[str, Any]) -> None:
        model_handle = cfg.get("model_handle")
        if not isinstance(model_handle, dict):
            raise TypeError("model_handle_builder.model_handle must be a dict.")
        raw_type = model_handle.get("type")
        if not isinstance(raw_type, str):
            raise TypeError("model_handle_builder.model_handle.type must be a string.")
        out_type = str(raw_type).strip().lower()
        if out_type == "":
            raise ValueError("model_handle_builder.model_handle.type cannot be empty.")
        handle_cfg = model_handle.get("config")
        if not isinstance(handle_cfg, dict):
            raise TypeError("model_handle_builder.model_handle.config must be a dict.")

        raw_subdir = handle_cfg.get("model_subdir")
        if not isinstance(raw_subdir, str):
            raise TypeError("model_handle_builder.model_handle.config.model_subdir must be a string.")
        model_subdir = str(raw_subdir).strip()
        if model_subdir == "":
            raise ValueError("model_handle_builder.model_handle.config.model_subdir cannot be empty.")

        raw = handle_cfg.get("model_path")
        if raw is not None and not isinstance(raw, str):
            raise TypeError("model_handle_builder.model_handle.config.model_path must be a string when provided.")
        model_path = None if raw is None else str(raw)

        cfg["model_handle"] = {
            "type": out_type,
            "config": {
                "model_path": model_path,
                "model_subdir": model_subdir,
            },
        }
