from __future__ import annotations

from ..base_pipeline_step import BasePipelineStep


class BaseWriterStep(BasePipelineStep):
    writer_name: str | None = None

    def default_config(self) -> dict:
        return {"config_json": {}}

    def build_writer_setup(
        self,
        *,
        output_dir: str | None,
        output_path: str | None,
    ) -> dict:
        cfg = self.get_config()
        config_json = dict(cfg.get("config_json") or {})
        writer_name = self.writer_name
        if writer_name is None:
            raise RuntimeError(f"{self.__class__.__name__} must define writer_name.")
        return {
            "writer_name": str(config_json.get("writer_name", writer_name)),
            "output_backend_name": str(config_json.get("output_backend_name", "parquet")),
            "output_dir": output_dir,
            "output_path": output_path,
            "fallback_output_dir": str(config_json.get("output_dir", "data/inference")),
            "streaming": bool(config_json.get("streaming", True)),
            "write_timestamped": bool(config_json.get("write_timestamped", False)),
        }

