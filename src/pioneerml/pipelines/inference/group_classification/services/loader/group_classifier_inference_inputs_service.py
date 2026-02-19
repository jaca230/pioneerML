from __future__ import annotations

from ..base import GroupClassifierInferenceServiceBase


class GroupClassifierInferenceInputsService(GroupClassifierInferenceServiceBase):
    step_key = "loader"

    def default_config(self) -> dict:
        return {"config_json": {}}

    def execute(self, *, parquet_paths: list[str]) -> dict:
        cfg = self.get_config()
        config_json = dict(cfg.get("config_json") or {})
        mode, batch_size, row_groups_per_chunk, num_workers = self.resolve_inference_runtime(config_json)
        resolved = self.resolve_paths(parquet_paths)

        return {
            "input_mode": "graph_loader_inference_v1",
            "mode": mode,
            "parquet_paths": resolved,
            "batch_size": int(batch_size),
            "row_groups_per_chunk": int(row_groups_per_chunk),
            "num_workers": int(num_workers),
            "num_rows": int(self.count_input_rows(resolved)),
            "validated_files": list(resolved),
        }
