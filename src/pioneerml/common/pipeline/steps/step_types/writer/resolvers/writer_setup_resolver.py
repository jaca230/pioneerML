from __future__ import annotations

from typing import Any

from pioneerml.common.pipeline.payloads import WriterSetupPayload

from ....resolver import BaseConfigResolver


class WriterSetupResolver(BaseConfigResolver):
    def resolve(self, *, cfg: dict[str, Any]) -> None:
        cfg["writer_setup_config"] = dict(cfg.get("config_json") or {})

    @staticmethod
    def resolve_writer_setup(
        *,
        cfg: dict,
        default_writer_name: str,
        output_dir: str | None,
        output_path: str | None,
    ) -> WriterSetupPayload:
        config_json = dict(cfg.get("config_json") or {})
        return WriterSetupPayload(
            writer_name=str(config_json.get("writer_name", default_writer_name)),
            output_backend_name=str(config_json.get("output_backend_name", "parquet")),
            output_dir=output_dir,
            output_path=output_path,
            fallback_output_dir=str(config_json.get("output_dir", "data/inference")),
            streaming=bool(config_json.get("streaming", True)),
            write_timestamped=bool(config_json.get("write_timestamped", False)),
            timestamp=(str(config_json["timestamp"]) if config_json.get("timestamp") is not None else None),
        )
