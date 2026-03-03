from pathlib import Path

import numpy as np
from zenml import step

from pioneerml.common.pipeline.steps import BaseTimeGroupOutputAdapterStep

from .inference_step import EndpointRegressorInferenceRunStep


class EndpointRegressorSavePredictionsStep(BaseTimeGroupOutputAdapterStep):
    step_key = "save_predictions"

    def default_config(self) -> dict:
        cfg = dict(super().default_config())
        cfg["output_dir"] = "data/endpoint_regressor"
        cfg["cleanup_streaming_tmp"] = True
        return cfg

    def execute(
        self,
        *,
        inference_outputs: dict,
        output_dir: str | None = None,
        output_path: str | None = None,
    ) -> dict:
        cfg = self.get_config()
        write_timestamped = bool(cfg.get("write_timestamped", False))
        cleanup_streaming_tmp = bool(cfg.get("cleanup_streaming_tmp", True))

        out_dir = self.ensure_output_dir(output_dir, str(cfg.get("output_dir", "data/endpoint_regressor")))
        ts = self.timestamp()

        validated_files = [str(Path(p).expanduser().resolve()) for p in inference_outputs.get("validated_files", [])]
        num_rows = int(inference_outputs.get("num_rows", 0))

        per_file_output_paths: list[str] = []
        per_file_timestamped_paths: list[str] = []

        streamed = list(inference_outputs.get("streamed_prediction_files") or [])
        if streamed:
            per_file_output_paths, per_file_timestamped_paths = self.promote_streamed_prediction_files(
                streamed_entries=streamed,
                validated_files=validated_files,
                output_dir=out_dir,
                output_path=output_path,
                write_timestamped=write_timestamped,
                timestamp=ts,
                cleanup_streaming_tmp=cleanup_streaming_tmp,
                streaming_tmp_dir=inference_outputs.get("streaming_tmp_dir"),
            )
        else:
            preds_np = inference_outputs["preds"].detach().cpu().numpy().astype(np.float32, copy=False)
            event_ids_np = inference_outputs["graph_event_id"].detach().cpu().numpy().astype(np.int64, copy=False)

            per_file_output_paths, per_file_timestamped_paths = self.write_non_streamed_time_group_predictions(
                output_dir=out_dir,
                output_path=output_path,
                write_timestamped=write_timestamped,
                timestamp=ts,
                validated_files=validated_files,
                event_ids_np=event_ids_np,
                prediction_columns=EndpointRegressorInferenceRunStep.prediction_columns_from_array(preds_np),
                value_types=EndpointRegressorInferenceRunStep.prediction_value_types(),
                num_rows=num_rows,
            )

        return {
            "predictions_path": per_file_output_paths[0] if len(per_file_output_paths) == 1 else None,
            "predictions_paths": per_file_output_paths,
            "timestamped_predictions_path": per_file_timestamped_paths[0] if len(per_file_timestamped_paths) == 1 else None,
            "timestamped_predictions_paths": per_file_timestamped_paths,
            "num_rows": num_rows,
        }


@step(name="save_endpoint_regressor_predictions", enable_cache=False)
def save_endpoint_regressor_predictions_step(
    inference_outputs: dict,
    output_dir: str | None = None,
    output_path: str | None = None,
    pipeline_config: dict | None = None,
) -> dict:
    return EndpointRegressorSavePredictionsStep(pipeline_config=pipeline_config).execute(
        inference_outputs=inference_outputs,
        output_dir=output_dir,
        output_path=output_path,
    )
