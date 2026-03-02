
from pathlib import Path

import numpy as np
from zenml import step

from pioneerml.common.pipeline.steps import BaseTimeGroupOutputAdapterStep


class GroupClassifierSavePredictionsStep(BaseTimeGroupOutputAdapterStep):
    step_key = "save_predictions"

    def default_config(self) -> dict:
        cfg = dict(super().default_config())
        cfg["output_dir"] = "data/group_classifier"
        cfg["cleanup_streaming_tmp"] = True
        cfg["metrics"] = ["binary_classification_from_counters", "binary_classification_from_tensors"]
        return cfg

    @staticmethod
    def _set_classification_metrics_none(metrics: dict) -> None:
        metrics["accuracy"] = None
        metrics["exact_match"] = None
        metrics["confusion"] = None

    def execute(
        self,
        *,
        inference_outputs: dict,
        output_dir: str | None = None,
        output_path: str | None = None,
        metrics_path: str | None = None,
    ) -> dict:
        cfg = self.get_config()
        write_timestamped = bool(cfg.get("write_timestamped", False))
        check_accuracy = bool(cfg.get("check_accuracy", False))
        cleanup_streaming_tmp = bool(cfg.get("cleanup_streaming_tmp", True))

        out_dir = self.ensure_output_dir(output_dir, str(cfg.get("output_dir", "data/group_classifier")))
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
            probs = inference_outputs["probs"]
            event_ids = inference_outputs["graph_event_ids"]
            probs_np = probs.detach().cpu().numpy().astype(np.float32, copy=False)
            event_ids_np = event_ids.detach().cpu().numpy().astype(np.int64, copy=False)
            per_file_output_paths, per_file_timestamped_paths = self.write_non_streamed_time_group_predictions(
                output_dir=out_dir,
                output_path=output_path,
                write_timestamped=write_timestamped,
                timestamp=ts,
                validated_files=validated_files,
                event_ids_np=event_ids_np,
                prediction_columns={
                    "pred_pion": probs_np[:, 0],
                    "pred_muon": probs_np[:, 1],
                    "pred_mip": probs_np[:, 2],
                },
                value_types=None,
                num_rows=num_rows,
            )

        metrics = {
            "mode": "group_classifier",
            "model_path": inference_outputs["model_path"],
            "output_path": per_file_output_paths[0] if len(per_file_output_paths) == 1 else None,
            "output_paths": per_file_output_paths,
            "threshold": float(inference_outputs["threshold"]),
            "validated_files": validated_files,
            "loss": None,
        }

        if check_accuracy:
            self.apply_registered_metrics(
                metrics=metrics,
                context={
                    "counters": inference_outputs.get("accuracy_counters"),
                    "preds_binary": inference_outputs.get("preds_binary"),
                    "targets": inference_outputs.get("targets"),
                },
            )
        else:
            self._set_classification_metrics_none(metrics)

        latest_metrics = Path(metrics_path) if metrics_path else out_dir / "metrics_latest.json"
        self.write_json(latest_metrics, metrics)

        timestamped_metrics_path = None
        if write_timestamped:
            p = out_dir / f"metrics_{ts}.json"
            self.write_json(p, metrics)
            timestamped_metrics_path = str(p)

        return {
            "predictions_path": per_file_output_paths[0] if len(per_file_output_paths) == 1 else None,
            "predictions_paths": per_file_output_paths,
            "metrics_path": str(latest_metrics),
            "timestamped_predictions_path": per_file_timestamped_paths[0] if len(per_file_timestamped_paths) == 1 else None,
            "timestamped_predictions_paths": per_file_timestamped_paths,
            "timestamped_metrics_path": timestamped_metrics_path,
            "num_rows": num_rows,
        }


@step(name="save_group_classifier_predictions", enable_cache=False)
def save_group_classifier_predictions_step(
    inference_outputs: dict,
    output_dir: str | None = None,
    output_path: str | None = None,
    metrics_path: str | None = None,
    pipeline_config: dict | None = None,
) -> dict:
    return GroupClassifierSavePredictionsStep(pipeline_config=pipeline_config).execute(
        inference_outputs=inference_outputs,
        output_dir=output_dir,
        output_path=output_path,
        metrics_path=metrics_path,
    )
