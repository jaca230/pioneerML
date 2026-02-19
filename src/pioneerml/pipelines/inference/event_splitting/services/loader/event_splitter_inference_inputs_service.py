from __future__ import annotations

from ..base import EventSplitterInferenceServiceBase


class EventSplitterInferenceInputsService(EventSplitterInferenceServiceBase):
    step_key = "loader"

    def default_config(self) -> dict:
        return {"config_json": {}}

    def execute(
        self,
        *,
        parquet_paths: list[str],
        group_probs_parquet_paths: list[str] | None,
        group_splitter_parquet_paths: list[str] | None,
        endpoint_parquet_paths: list[str] | None,
    ) -> dict:
        cfg = self.get_config()
        config_json = dict(cfg.get("config_json") or {})
        (
            mode,
            use_group_probs,
            use_splitter_probs,
            use_endpoint_preds,
            batch_size,
            row_groups_per_chunk,
            num_workers,
        ) = self.resolve_inference_runtime(config_json)
        resolved = self.resolve_paths(parquet_paths)
        resolved_group_probs = self.resolve_optional_paths(group_probs_parquet_paths)
        resolved_splitter_probs = self.resolve_optional_paths(group_splitter_parquet_paths)
        resolved_endpoints = self.resolve_optional_paths(endpoint_parquet_paths)

        return {
            "input_mode": "event_splitter_graph_loader_inference_v1",
            "mode": mode,
            "use_group_probs": bool(use_group_probs),
            "use_splitter_probs": bool(use_splitter_probs),
            "use_endpoint_preds": bool(use_endpoint_preds),
            "parquet_paths": resolved,
            "group_probs_parquet_paths": resolved_group_probs,
            "group_splitter_parquet_paths": resolved_splitter_probs,
            "endpoint_parquet_paths": resolved_endpoints,
            "batch_size": int(batch_size),
            "row_groups_per_chunk": int(row_groups_per_chunk),
            "num_workers": int(num_workers),
            "num_rows": int(self.count_input_rows(resolved)),
            "validated_files": list(resolved),
            "validated_group_probs_files": list(resolved_group_probs or []),
            "validated_group_splitter_files": list(resolved_splitter_probs or []),
            "validated_endpoint_files": list(resolved_endpoints or []),
        }
