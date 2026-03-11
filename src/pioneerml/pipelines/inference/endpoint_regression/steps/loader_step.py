from zenml import step

from pioneerml.common.pipeline.steps import BaseLoaderFactoryInitStep


class EndpointRegressorInferenceInputsStep(BaseLoaderFactoryInitStep):
    step_key = "loader"

    def default_config(self) -> dict:
        return {"config_json": {}}

    def run(
        self,
        *,
        input_source_set: dict,
    ) -> dict:
        cfg = self.config_json
        config_json = dict(cfg.get("config_json") or {})
        mode, data_flow_config = self.resolve_loader_runtime(
            config_json,
            default_mode="inference",
            allowed_modes=("inference", "train"),
            default_batch_size=64,
            default_chunk_row_groups=4,
        )
        input_sources = self.resolve_input_source_set(input_source_set)
        resolved = list(input_sources.main_sources)
        input_backend_name = str(config_json.get("input_backend_name", "parquet"))
        validated_file_rows = self.count_source_rows_per_file(
            input_sources=input_sources,
            input_backend_name=input_backend_name,
        )
        group_probs_paths = input_sources.source_entries("group_probs")
        resolved_group_probs = list(group_probs_paths) if group_probs_paths is not None else None
        splitter_paths = input_sources.source_entries("group_splitter")
        resolved_splitter_probs = list(splitter_paths) if splitter_paths is not None else None

        return {
            "input_mode": "endpoint_regressor_graph_loader_inference_v1",
            "mode": mode,
            "input_backend_name": input_backend_name,
            "main_sources": resolved,
            "group_probs_sources": resolved_group_probs,
            "group_splitter_sources": resolved_splitter_probs,
            "data_flow_config": {
                "batch_size": int(data_flow_config.batch_size),
                "row_groups_per_chunk": int(data_flow_config.row_groups_per_chunk),
                "num_workers": int(data_flow_config.num_workers),
            },
            "batch_size": int(data_flow_config.batch_size),
            "row_groups_per_chunk": int(data_flow_config.row_groups_per_chunk),
            "num_workers": int(data_flow_config.num_workers),
            "num_rows": int(sum(validated_file_rows)),
            "validated_file_rows": validated_file_rows,
            "validated_files": list(resolved),
            "validated_group_probs_sources": list(resolved_group_probs or []),
            "validated_group_splitter_sources": list(resolved_splitter_probs or []),
        }


@step(name="load_endpoint_regressor_inference_inputs", enable_cache=False)
def load_endpoint_regressor_inference_inputs_step(
    input_source_set: dict,
    pipeline_config: dict | None = None,
) -> dict:
    return EndpointRegressorInferenceInputsStep(pipeline_config=pipeline_config).execute(
        input_source_set=input_source_set,
    )
