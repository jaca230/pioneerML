from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from pioneerml.data_writer import BaseDataWriter, WriterFactory, WriterRunConfig
from pioneerml.data_writer.backends import create_output_backend

from ......resolver import BaseConfigResolver


class InferenceConfigResolver(BaseConfigResolver):
    def resolve(self, *, cfg: dict[str, Any]) -> None:
        runtime_cfg = cfg.get("runtime")
        if runtime_cfg is None:
            runtime_cfg = {}
        if not isinstance(runtime_cfg, dict):
            raise TypeError("inference.runtime must be a dict when provided.")
        cfg["runtime"] = {
            "prefer_cuda": bool(runtime_cfg.get("prefer_cuda", True)),
        }
        writer = cfg.get("writer")
        if not isinstance(writer, dict):
            raise TypeError("inference.writer must be a dict.")
        writer_type = writer.get("type")
        if not isinstance(writer_type, str) or writer_type.strip() == "":
            raise ValueError("inference.writer.type must be a non-empty string.")
        if str(writer_type).strip().lower() == "required":
            raise ValueError("inference.writer.type must be set to a concrete registered writer plugin.")
        writer_cfg = writer.get("config")
        if not isinstance(writer_cfg, dict):
            raise TypeError("inference.writer.config must be a dict.")

        output_backend = self._normalize_output_backend(
            writer_cfg=writer_cfg,
            context="inference.writer.config.output_backend",
        )

        fallback_output_dir = writer_cfg.get("fallback_output_dir", "data/inference")
        if not isinstance(fallback_output_dir, str):
            raise TypeError("inference.writer.config.fallback_output_dir must be a string.")
        fallback_output_dir = fallback_output_dir.strip()
        if not fallback_output_dir:
            raise ValueError("inference.writer.config.fallback_output_dir cannot be empty.")

        output_dir = writer_cfg.get("output_dir")
        if output_dir is not None and not isinstance(output_dir, str):
            raise TypeError("inference.writer.config.output_dir must be a string when provided.")
        output_dir = (None if output_dir is None else str(output_dir))

        output_path = writer_cfg.get("output_path")
        if output_path is not None and not isinstance(output_path, str):
            raise TypeError("inference.writer.config.output_path must be a string when provided.")
        output_path = (None if output_path is None else str(output_path))

        streaming = bool(writer_cfg.get("streaming", True))
        write_timestamped = bool(writer_cfg.get("write_timestamped", False))

        timestamp = writer_cfg.get("timestamp")
        if timestamp is not None and not isinstance(timestamp, str):
            raise TypeError("inference.writer.config.timestamp must be a string when provided.")
        timestamp = (None if timestamp is None else str(timestamp))

        writer_params = writer_cfg.get("writer_params", {})
        if not isinstance(writer_params, dict):
            raise TypeError("inference.writer.config.writer_params must be a dict when provided.")
        writer_params = dict(writer_params)

        cfg["writer"] = {
            "type": str(writer_type).strip(),
            "config": {
                "output_backend": output_backend,
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
        output_backend_block = dict(writer_cfg.get("output_backend") or {})
        output_backend_name = str(output_backend_block["type"]).strip().lower()
        output_backend_cfg = dict(output_backend_block.get("config") or {})
        output_backend = create_output_backend(output_backend_name, config=output_backend_cfg)
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
                "output_backend_name": output_backend_name,
                "run_config": run_config,
                "writer_params": {
                    **dict(writer_cfg.get("writer_params") or {}),
                    "output_backend": output_backend,
                },
                "output_path": (output_path if output_path is not None else writer_cfg.get("output_path")),
            },
        )

    @staticmethod
    def _normalize_output_backend(*, writer_cfg: Mapping[str, Any], context: str) -> dict[str, Any]:
        raw_backend = writer_cfg.get("output_backend")
        if not isinstance(raw_backend, Mapping):
            raise TypeError(f"{context} must be a mapping with keys ['type', 'config'].")
        backend_block = dict(raw_backend)
        backend_type = backend_block.get("type")
        if not isinstance(backend_type, str) or backend_type.strip() == "":
            raise TypeError(f"{context}.type must be a non-empty string.")
        backend_cfg = backend_block.get("config")
        if backend_cfg is None:
            backend_cfg = {}
        if not isinstance(backend_cfg, Mapping):
            raise TypeError(f"{context}.config must be a mapping.")
        backend_cfg = dict(backend_cfg)

        target_row_group_rows = backend_cfg.get("target_row_group_rows", 1024)
        if target_row_group_rows is None:
            target_row_group_rows = 1024
        target_row_group_rows = int(target_row_group_rows)
        if target_row_group_rows <= 0:
            raise ValueError(f"{context}.config.target_row_group_rows must be positive.")
        backend_cfg["target_row_group_rows"] = target_row_group_rows

        return {
            "type": str(backend_type).strip().lower(),
            "config": backend_cfg,
        }
