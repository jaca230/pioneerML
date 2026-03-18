from __future__ import annotations

from typing import Any

from .....resolver import BaseConfigResolver


class ModelHandleBuilderConfigResolver(BaseConfigResolver):
    def resolve(self, *, cfg: dict[str, Any]) -> None:
        raw_subdir = cfg.get("model_subdir")
        if not isinstance(raw_subdir, str):
            raise TypeError("model_handle_builder.model_subdir must be a string.")
        model_subdir = str(raw_subdir).strip()
        if model_subdir == "":
            raise ValueError("model_handle_builder.model_subdir cannot be empty.")
        cfg["model_subdir"] = model_subdir

        raw = cfg.get("model_path")
        if raw is not None and not isinstance(raw, str):
            raise TypeError("model_handle_builder.model_path must be a string when provided.")
        cfg["model_path"] = None if raw is None else str(raw)

        raw_type = cfg.get("model_type", "script")
        if not isinstance(raw_type, str):
            raise TypeError("model_handle_builder.model_type must be a string when provided.")
        out_type = str(raw_type).strip().lower()
        if out_type == "":
            raise ValueError("model_handle_builder.model_type cannot be empty.")
        cfg["model_type"] = out_type
