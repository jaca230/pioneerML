from __future__ import annotations

from pathlib import Path

import pyarrow.parquet as pq
import torch
from pioneerml.common.pipeline.services import BaseOutputAdapterService

from ..base import EndpointRegressorInferenceServiceBase


class EndpointRegressorSavePredictionsService(
    EndpointRegressorInferenceServiceBase,
    BaseOutputAdapterService,
):
    step_key = "save_predictions"

    def default_config(self) -> dict:
        cfg = dict(BaseOutputAdapterService.default_config(self))
        cfg["output_dir"] = "data/endpoint_regressor"
        cfg["cleanup_streaming_tmp"] = True
        return cfg

    @staticmethod
    def _apply_regression_metrics_from_counters(metrics: dict, counters: dict | None) -> bool:
        if not counters or not bool(counters.get("has_targets", False)):
            metrics["loss"] = None
            metrics["mae"] = None
            return False
        count = int(counters.get("count", 0))
        if count <= 0:
            metrics["loss"] = None
            metrics["mae"] = None
            return True
        metrics["loss"] = float(counters.get("sum_sq_error", 0.0)) / float(count)
        metrics["mae"] = float(counters.get("sum_abs_error", 0.0)) / float(count)
        return True

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

        out_dir = self.ensure_output_dir(output_dir, str(cfg.get("output_dir", "data/endpoint_regressor")))

        targets = inference_outputs.get("targets")
        num_rows = int(inference_outputs["num_rows"])
        prediction_dim = inference_outputs.get("prediction_dim")

        ts = self.timestamp()
        validated_files = [str(Path(p).expanduser().resolve()) for p in inference_outputs.get("validated_files", [])]
        validated_group_probs_files = [
            str(Path(p).expanduser().resolve()) for p in inference_outputs.get("validated_group_probs_files", [])
        ]
        validated_group_splitter_files = [
            str(Path(p).expanduser().resolve()) for p in inference_outputs.get("validated_group_splitter_files", [])
        ]
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
            preds = inference_outputs["preds"]
            event_ids = inference_outputs["graph_event_ids"]
            group_ids = inference_outputs["graph_group_ids"]
            preds_np = preds.detach().cpu().numpy().astype("float32", copy=False)
            event_ids_np = event_ids.detach().cpu().numpy().astype("int64", copy=False)
            group_ids_np = group_ids.detach().cpu().numpy().astype("int64", copy=False)
            prediction_dim = int(preds_np.shape[1]) if preds_np.ndim == 2 else prediction_dim

            if output_path and len(validated_files) != 1:
                raise ValueError("output_path is only supported when exactly one input parquet file is provided.")

            if validated_files:
                row_counts = [int(pq.ParquetFile(p).metadata.num_rows) for p in validated_files]
                start = 0
                for src_file, n_rows in zip(validated_files, row_counts, strict=True):
                    end = start + n_rows
                    mask = (event_ids_np >= start) & (event_ids_np < end)
                    local_event_ids = event_ids_np[mask] - start
                    local_group_ids = group_ids_np[mask]
                    local_preds = preds_np[mask]
                    table = self.build_prediction_table(
                        event_ids_np=local_event_ids,
                        group_ids_np=local_group_ids,
                        preds_np=local_preds,
                        num_rows=n_rows,
                    )

                    pred_path = (
                        Path(output_path)
                        if (output_path and len(validated_files) == 1)
                        else out_dir / f"{Path(src_file).stem}_preds.parquet"
                    )
                    self.atomic_write_table(table=table, dst_path=pred_path)
                    per_file_output_paths.append(str(pred_path))

                    if write_timestamped:
                        p = out_dir / f"{Path(src_file).stem}_preds_{ts}.parquet"
                        self.atomic_write_table(table=table, dst_path=p)
                        per_file_timestamped_paths.append(str(p))
                    start = end
            else:
                table = self.build_prediction_table(
                    event_ids_np=event_ids_np,
                    group_ids_np=group_ids_np,
                    preds_np=preds_np,
                    num_rows=num_rows,
                )
                pred_path = Path(output_path) if output_path else (out_dir / "endpoint_regressor_preds.parquet")
                self.atomic_write_table(table=table, dst_path=pred_path)
                per_file_output_paths.append(str(pred_path))
                if write_timestamped:
                    p = out_dir / f"endpoint_regressor_preds_{ts}.parquet"
                    self.atomic_write_table(table=table, dst_path=p)
                    per_file_timestamped_paths.append(str(p))

        metrics = {
            "mode": "endpoint_regressor",
            "model_path": inference_outputs["model_path"],
            "output_path": per_file_output_paths[0] if len(per_file_output_paths) == 1 else None,
            "output_paths": per_file_output_paths,
            "prediction_dim": int(prediction_dim) if prediction_dim is not None else None,
            "validated_files": validated_files,
            "validated_group_probs_files": validated_group_probs_files,
            "validated_group_splitter_files": validated_group_splitter_files,
            "loss": None,
            "mae": None,
        }

        if check_accuracy:
            counters = inference_outputs.get("regression_counters")
            used_counters = self._apply_regression_metrics_from_counters(metrics, counters)
            if (not used_counters) and targets is not None and targets.numel() > 0:
                preds_t = inference_outputs["preds"].to(torch.float32)
                targets_t = targets.to(torch.float32)
                mse = torch.mean((preds_t - targets_t) ** 2)
                mae = torch.mean(torch.abs(preds_t - targets_t))
                metrics["loss"] = float(mse.item())
                metrics["mae"] = float(mae.item())

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
            "timestamped_predictions_path": per_file_timestamped_paths[0]
            if len(per_file_timestamped_paths) == 1
            else None,
            "timestamped_predictions_paths": per_file_timestamped_paths,
            "timestamped_metrics_path": timestamped_metrics_path,
            "num_rows": num_rows,
        }
