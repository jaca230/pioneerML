from __future__ import annotations

from pioneerml.common.data_writer import BaseDataWriter, WriterFactory, WriterRunConfig
from typing import Any

from .....resolver import BaseConfigResolver


class WriterRuntimeConfigResolver(BaseConfigResolver):
    def resolve(self, *, cfg: dict[str, Any]) -> None:
        writer_name = cfg.get("writer_name")
        if not isinstance(writer_name, str):
            raise TypeError("writer.writer_name must be a string.")
        writer_name = writer_name.strip()
        if writer_name == "":
            raise ValueError("writer.writer_name cannot be empty.")
        cfg["writer_name"] = writer_name

        output_backend_name = cfg.get("output_backend_name", "parquet")
        if not isinstance(output_backend_name, str):
            raise TypeError("writer.output_backend_name must be a string.")
        cfg["output_backend_name"] = str(output_backend_name).strip().lower() or "parquet"

        fallback_output_dir = cfg.get("fallback_output_dir", "data/inference")
        if not isinstance(fallback_output_dir, str):
            raise TypeError("writer.fallback_output_dir must be a string.")
        fallback_output_dir = fallback_output_dir.strip()
        if not fallback_output_dir:
            raise ValueError("writer.fallback_output_dir cannot be empty.")
        cfg["fallback_output_dir"] = fallback_output_dir

        output_dir = cfg.get("output_dir")
        if output_dir is not None and not isinstance(output_dir, str):
            raise TypeError("writer.output_dir must be a string when provided.")
        cfg["output_dir"] = (None if output_dir is None else str(output_dir))

        output_path = cfg.get("output_path")
        if output_path is not None and not isinstance(output_path, str):
            raise TypeError("writer.output_path must be a string when provided.")
        cfg["output_path"] = (None if output_path is None else str(output_path))

        cfg["streaming"] = bool(cfg.get("streaming", True))
        cfg["write_timestamped"] = bool(cfg.get("write_timestamped", False))

        timestamp = cfg.get("timestamp")
        if timestamp is not None and not isinstance(timestamp, str):
            raise TypeError("writer.timestamp must be a string when provided.")
        cfg["timestamp"] = (None if timestamp is None else str(timestamp))

        writer_params = cfg.get("writer_params", {})
        if not isinstance(writer_params, dict):
            raise TypeError("writer.writer_params must be a dict when provided.")
        cfg["writer_params"] = dict(writer_params)
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
        resolved_output_dir = BaseDataWriter.ensure_output_dir(
            (output_dir if output_dir is not None else cfg.get("output_dir")),
            str(cfg["fallback_output_dir"]),
        )
        timestamp = cfg.get("timestamp")
        run_config = WriterRunConfig(
            output_dir=resolved_output_dir,
            timestamp=(str(timestamp) if timestamp is not None else BaseDataWriter.timestamp()),
            streaming=bool(cfg["streaming"]),
            write_timestamped=bool(cfg["write_timestamped"]),
        )
        return WriterFactory(
            writer_name=str(cfg["writer_name"]),
            output_backend_name=str(cfg["output_backend_name"]),
            run_config=run_config,
            writer_params=dict(cfg.get("writer_params") or {}),
            output_path=(output_path if output_path is not None else cfg.get("output_path")),
        )
