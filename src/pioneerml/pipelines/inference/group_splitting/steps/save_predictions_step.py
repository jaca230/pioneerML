
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from zenml import step

from pioneerml.common.pipeline.steps import BaseTimeGroupOutputAdapterStep


class GroupSplitterSavePredictionsStep(BaseTimeGroupOutputAdapterStep):
    step_key = "save_predictions"

    def default_config(self) -> dict:
        cfg = dict(super().default_config())
        cfg["output_dir"] = "data/group_splitter"
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

        out_dir = self.ensure_output_dir(output_dir, str(cfg.get("output_dir", "data/group_splitter")))
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
            probs_np = inference_outputs["probs"].detach().cpu().numpy().astype(np.float32, copy=False)
            node_event_ids_np = inference_outputs["node_event_ids"].detach().cpu().numpy().astype(np.int64, copy=False)
            graph_event_ids_np = inference_outputs["graph_event_id"].detach().cpu().numpy().astype(np.int64, copy=False)
            graph_group_ids_np = inference_outputs["graph_time_group_id"].detach().cpu().numpy().astype(np.int64, copy=False)

            if output_path and len(validated_files) != 1:
                raise ValueError("output_path is only supported when exactly one input parquet file is provided.")

            if validated_files:
                row_counts = [int(pq.ParquetFile(p).metadata.num_rows) for p in validated_files]
                start = 0
                for src_file, n_rows in zip(validated_files, row_counts, strict=True):
                    end = start + n_rows
                    node_mask = (node_event_ids_np >= start) & (node_event_ids_np < end)
                    graph_mask = (graph_event_ids_np >= start) & (graph_event_ids_np < end)
                    table = self.stitch_node_predictions_to_events(
                        node_event_ids_np=(node_event_ids_np[node_mask] - start),
                        graph_event_ids_np=(graph_event_ids_np[graph_mask] - start),
                        graph_group_ids_np=graph_group_ids_np[graph_mask],
                        prediction_columns={
                            "pred_hit_pion": probs_np[node_mask, 0],
                            "pred_hit_muon": probs_np[node_mask, 1],
                            "pred_hit_mip": probs_np[node_mask, 2],
                        },
                        num_rows=n_rows,
                    )
                    pred_path = (
                        Path(output_path).expanduser().resolve()
                        if (output_path and len(validated_files) == 1)
                        else out_dir / f"{Path(src_file).stem}_preds.parquet"
                    )
                    self.atomic_write_table(table=table, dst_path=pred_path)
                    per_file_output_paths.append(str(pred_path))
                    if write_timestamped:
                        ts_path = out_dir / f"{Path(src_file).stem}_preds_{ts}.parquet"
                        self.atomic_write_table(table=table, dst_path=ts_path)
                        per_file_timestamped_paths.append(str(ts_path))
                    start = end
            else:
                table = self.stitch_node_predictions_to_events(
                    node_event_ids_np=node_event_ids_np,
                    graph_event_ids_np=graph_event_ids_np,
                    graph_group_ids_np=graph_group_ids_np,
                    prediction_columns={
                        "pred_hit_pion": probs_np[:, 0],
                        "pred_hit_muon": probs_np[:, 1],
                        "pred_hit_mip": probs_np[:, 2],
                    },
                    num_rows=num_rows,
                )
                pred_path = Path(output_path).expanduser().resolve() if output_path else (out_dir / "preds.parquet")
                self.atomic_write_table(table=table, dst_path=pred_path)
                per_file_output_paths.append(str(pred_path))
                if write_timestamped:
                    ts_path = out_dir / f"preds_{ts}.parquet"
                    self.atomic_write_table(table=table, dst_path=ts_path)
                    per_file_timestamped_paths.append(str(ts_path))

        return {
            "predictions_path": per_file_output_paths[0] if len(per_file_output_paths) == 1 else None,
            "predictions_paths": per_file_output_paths,
            "timestamped_predictions_path": per_file_timestamped_paths[0] if len(per_file_timestamped_paths) == 1 else None,
            "timestamped_predictions_paths": per_file_timestamped_paths,
            "num_rows": num_rows,
        }


@step(name="save_group_splitter_predictions", enable_cache=False)
def save_group_splitter_predictions_step(
    inference_outputs: dict,
    output_dir: str | None = None,
    output_path: str | None = None,
    pipeline_config: dict | None = None,
) -> dict:
    return GroupSplitterSavePredictionsStep(pipeline_config=pipeline_config).execute(
        inference_outputs=inference_outputs,
        output_dir=output_dir,
        output_path=output_path,
    )
