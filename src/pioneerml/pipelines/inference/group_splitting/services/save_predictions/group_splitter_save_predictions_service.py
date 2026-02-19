from __future__ import annotations

from pathlib import Path

import pyarrow.parquet as pq
import torch
from pioneerml.common.pipeline.services import BaseOutputAdapterService

from ..base import GroupSplitterInferenceServiceBase


class GroupSplitterSavePredictionsService(GroupSplitterInferenceServiceBase, BaseOutputAdapterService):
    step_key = "save_predictions"

    def default_config(self) -> dict:
        cfg = dict(BaseOutputAdapterService.default_config(self))
        cfg["output_dir"] = "data/group_splitter"
        cfg["cleanup_streaming_tmp"] = True
        return cfg

    @staticmethod
    def _apply_accuracy_from_counters(metrics: dict, counters: dict | None) -> None:
        if not counters or not bool(counters.get("has_targets", False)):
            metrics["accuracy"] = None
            metrics["exact_match"] = None
            metrics["confusion"] = None
            return

        label_total = int(counters.get("label_total", 0))
        label_equal = int(counters.get("label_equal", 0))
        graph_total = int(counters.get("graph_total", 0))
        graph_exact = int(counters.get("graph_exact", 0))
        tp = [int(v) for v in counters.get("tp", [])]
        fp = [int(v) for v in counters.get("fp", [])]
        fn = [int(v) for v in counters.get("fn", [])]

        metrics["accuracy"] = (float(label_equal) / float(label_total)) if label_total > 0 else None
        metrics["exact_match"] = (float(graph_exact) / float(graph_total)) if graph_total > 0 else None

        confusion: list[dict[str, float]] = []
        for i in range(min(len(tp), len(fp), len(fn))):
            total = float(tp[i] + fp[i] + fn[i])
            if total > 0:
                confusion.append({"tp": tp[i] / total, "fp": fp[i] / total, "fn": fn[i] / total})
            else:
                confusion.append({"tp": 0.0, "fp": 0.0, "fn": 0.0})
        metrics["confusion"] = confusion

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

        out_dir = self.ensure_output_dir(output_dir, str(cfg.get("output_dir", "data/group_splitter")))

        num_rows = int(inference_outputs["num_rows"])
        targets = inference_outputs.get("targets")

        ts = self.timestamp()
        validated_files = [str(Path(p).expanduser().resolve()) for p in inference_outputs.get("validated_files", [])]
        validated_group_probs_files = [
            str(Path(p).expanduser().resolve()) for p in inference_outputs.get("validated_group_probs_files", [])
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
            probs = inference_outputs["probs"]
            event_ids = inference_outputs["node_event_ids"]
            time_group_ids = inference_outputs["node_time_group_ids"]
            probs_np = probs.detach().cpu().numpy().astype("float32", copy=False)
            event_ids_np = event_ids.detach().cpu().numpy().astype("int64", copy=False)
            tg_np = time_group_ids.detach().cpu().numpy().astype("int64", copy=False)

            if output_path and len(validated_files) != 1:
                raise ValueError("output_path is only supported when exactly one input parquet file is provided.")

            if validated_files:
                row_counts = [int(pq.ParquetFile(p).metadata.num_rows) for p in validated_files]
                start = 0
                for src_file, n_rows in zip(validated_files, row_counts, strict=True):
                    end = start + n_rows
                    mask = (event_ids_np >= start) & (event_ids_np < end)
                    local_event_ids = event_ids_np[mask] - start
                    local_tg = tg_np[mask]
                    local_probs = probs_np[mask]
                    table = self.build_prediction_table(
                        event_ids_np=local_event_ids,
                        time_group_ids_np=local_tg,
                        probs_np=local_probs,
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
                    time_group_ids_np=tg_np,
                    probs_np=probs_np,
                    num_rows=num_rows,
                )
                pred_path = Path(output_path) if output_path else (out_dir / "group_splitter_preds.parquet")
                self.atomic_write_table(table=table, dst_path=pred_path)
                per_file_output_paths.append(str(pred_path))
                if write_timestamped:
                    p = out_dir / f"group_splitter_preds_{ts}.parquet"
                    self.atomic_write_table(table=table, dst_path=p)
                    per_file_timestamped_paths.append(str(p))

        metrics = {
            "mode": "group_splitter",
            "model_path": inference_outputs["model_path"],
            "output_path": per_file_output_paths[0] if len(per_file_output_paths) == 1 else None,
            "output_paths": per_file_output_paths,
            "threshold": float(inference_outputs["threshold"]),
            "validated_files": validated_files,
            "validated_group_probs_files": validated_group_probs_files,
            "loss": None,
        }

        if check_accuracy:
            counters = inference_outputs.get("accuracy_counters")
            if counters is not None:
                self._apply_accuracy_from_counters(metrics, counters)
            elif targets is not None and targets.numel() > 0:
                preds_binary = inference_outputs["preds_binary"].to(torch.float32)
                t = targets.to(torch.float32)
                metrics["accuracy"] = float((preds_binary == t).to(torch.float32).mean().item())
                metrics["exact_match"] = float(((preds_binary == t).all(dim=1)).to(torch.float32).mean().item())
                confusion = []
                for cls in range(int(t.shape[1])):
                    truth = t[:, cls].to(torch.int64)
                    pred = preds_binary[:, cls].to(torch.int64)
                    fp = int(((truth == 0) & (pred == 1)).sum().item())
                    fn = int(((truth == 1) & (pred == 0)).sum().item())
                    tp = int(((truth == 1) & (pred == 1)).sum().item())
                    total = float(tp + fp + fn)
                    if total > 0:
                        confusion.append({"tp": tp / total, "fp": fp / total, "fn": fn / total})
                    else:
                        confusion.append({"tp": 0.0, "fp": 0.0, "fn": 0.0})
                metrics["confusion"] = confusion
            else:
                metrics["accuracy"] = None
                metrics["exact_match"] = None
                metrics["confusion"] = None
        else:
            metrics["accuracy"] = None
            metrics["exact_match"] = None
            metrics["confusion"] = None

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
