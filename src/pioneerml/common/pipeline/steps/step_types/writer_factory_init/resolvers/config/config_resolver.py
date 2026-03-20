from __future__ import annotations

from pioneerml.common.data_writer import BaseDataWriter, WriterFactory, WriterRunConfig
from typing import Any

from .....resolver import BaseConfigResolver


class WriterConfigResolver(BaseConfigResolver):
    def resolve(self, *, cfg: dict[str, Any]) -> None:
        writer = cfg.get("writer")
        if not isinstance(writer, dict):
            raise TypeError("writer.writer must be a dict.")
        writer_type = writer.get("type")
        if not isinstance(writer_type, str) or writer_type.strip() == "":
            raise ValueError("writer.writer.type must be a non-empty string.")
        if str(writer_type).strip().lower() == "required":
            raise ValueError("writer.writer.type must be set to a concrete registered writer plugin.")
        writer_cfg = writer.get("config")
        if not isinstance(writer_cfg, dict):
            raise TypeError("writer.writer.config must be a dict.")

        output_backend_name = writer_cfg.get("output_backend_name", "parquet")
        if not isinstance(output_backend_name, str):
            raise TypeError("writer.writer.config.output_backend_name must be a string.")
        output_backend_name = str(output_backend_name).strip().lower() or "parquet"

        fallback_output_dir = writer_cfg.get("fallback_output_dir", "data/inference")
        if not isinstance(fallback_output_dir, str):
            raise TypeError("writer.writer.config.fallback_output_dir must be a string.")
        fallback_output_dir = fallback_output_dir.strip()
        if not fallback_output_dir:
            raise ValueError("writer.writer.config.fallback_output_dir cannot be empty.")

        output_dir = writer_cfg.get("output_dir")
        if output_dir is not None and not isinstance(output_dir, str):
            raise TypeError("writer.writer.config.output_dir must be a string when provided.")
        output_dir = (None if output_dir is None else str(output_dir))

        output_path = writer_cfg.get("output_path")
        if output_path is not None and not isinstance(output_path, str):
            raise TypeError("writer.writer.config.output_path must be a string when provided.")
        output_path = (None if output_path is None else str(output_path))

        streaming = bool(writer_cfg.get("streaming", True))
        write_timestamped = bool(writer_cfg.get("write_timestamped", False))

        timestamp = writer_cfg.get("timestamp")
        if timestamp is not None and not isinstance(timestamp, str):
            raise TypeError("writer.writer.config.timestamp must be a string when provided.")
        timestamp = (None if timestamp is None else str(timestamp))

        writer_params = writer_cfg.get("writer_params", {})
        if not isinstance(writer_params, dict):
            raise TypeError("writer.writer.config.writer_params must be a dict when provided.")
        writer_params = dict(writer_params)

        cfg["writer"] = {
            "type": str(writer_type).strip(),
            "config": {
                "output_backend_name": output_backend_name,
                "fallback_output_dir": fallback_output_dir,
                "output_dir": output_dir,
                "output_path": output_path,
                "streaming": streaming,
                "write_timestamped": write_timestamped,
                "timestamp": timestamp,
                "writer_params": writer_params,
            },
        }
        self.step.runtime_state["writer_factory"] = self.build_writer_factory(
            cfg=cfg,
            output_dir=None,
            output_path=None,
        )

    @staticmethod
    def build_writer_factory(
        *,
        cfg: dict[str, Any],
        output_dir: str | None,
        output_path: str | None,
    ) -> WriterFactory:
        writer = dict(cfg["writer"])
        writer_cfg = dict(writer["config"])
        resolved_output_dir = BaseDataWriter.ensure_output_dir(
            (output_dir if output_dir is not None else writer_cfg.get("output_dir")),
            str(writer_cfg["fallback_output_dir"]),
        )
        timestamp = writer_cfg.get("timestamp")
        run_config = WriterRunConfig(
            output_dir=resolved_output_dir,
            timestamp=(str(timestamp) if timestamp is not None else BaseDataWriter.timestamp()),
            streaming=bool(writer_cfg["streaming"]),
            write_timestamped=bool(writer_cfg["write_timestamped"]),
        )
        return WriterFactory(
            writer_name=str(writer["type"]),
            config={
                "output_backend_name": str(writer_cfg["output_backend_name"]),
                "run_config": run_config,
                "writer_params": dict(writer_cfg.get("writer_params") or {}),
                "output_path": (output_path if output_path is not None else writer_cfg.get("output_path")),
            },
        )
