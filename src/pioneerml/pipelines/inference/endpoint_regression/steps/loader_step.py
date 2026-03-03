from zenml import step

from pioneerml.common.pipeline.steps import BaseLoaderStep


class EndpointRegressorInferenceInputsStep(BaseLoaderStep):
    step_key = "loader"

    def default_config(self) -> dict:
        return {"config_json": {}}

    def execute(
        self,
        *,
        parquet_input_set: dict,
    ) -> dict:
        cfg = self.get_config()
        config_json = dict(cfg.get("config_json") or {})
        mode, data_flow_config = self.resolve_parquet_runtime(
            config_json,
            default_mode="inference",
            allowed_modes=("inference", "train"),
            default_batch_size=64,
            default_chunk_row_groups=4,
        )
        parquet_inputs = self.resolve_parquet_input_set(parquet_input_set)
        resolved = list(parquet_inputs.main_paths)
        group_probs_paths = parquet_inputs.source_paths("group_probs")
        resolved_group_probs = list(group_probs_paths) if group_probs_paths is not None else None
        splitter_paths = parquet_inputs.source_paths("group_splitter")
        resolved_splitter_probs = list(splitter_paths) if splitter_paths is not None else None

        return {
            "input_mode": "endpoint_regressor_graph_loader_inference_v1",
            "mode": mode,
            "parquet_paths": resolved,
            "group_probs_parquet_paths": resolved_group_probs,
            "group_splitter_parquet_paths": resolved_splitter_probs,
            "data_flow_config": {
                "batch_size": int(data_flow_config.batch_size),
                "row_groups_per_chunk": int(data_flow_config.row_groups_per_chunk),
                "num_workers": int(data_flow_config.num_workers),
            },
            "batch_size": int(data_flow_config.batch_size),
            "row_groups_per_chunk": int(data_flow_config.row_groups_per_chunk),
            "num_workers": int(data_flow_config.num_workers),
            "num_rows": int(self.count_parquet_rows(resolved)),
            "validated_files": list(resolved),
            "validated_group_probs_files": list(resolved_group_probs or []),
            "validated_group_splitter_files": list(resolved_splitter_probs or []),
        }


@step(name="load_endpoint_regressor_inference_inputs", enable_cache=False)
def load_endpoint_regressor_inference_inputs_step(
    parquet_input_set: dict,
    pipeline_config: dict | None = None,
) -> dict:
    return EndpointRegressorInferenceInputsStep(pipeline_config=pipeline_config).execute(
        parquet_input_set=parquet_input_set,
    )
