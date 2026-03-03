
from pathlib import Path

import torch
from zenml import step

from pioneerml.common.loader import GroupSplitterGraphLoader
from pioneerml.common.pipeline.steps import BaseInferenceStep, BaseTimeGroupOutputAdapterStep


class GroupSplitterInferenceRunStep(BaseInferenceStep):
    step_key = "inference"

    def default_config(self) -> dict:
        return {
            "threshold": 0.5,
            "streaming": True,
            "materialize_outputs": None,
            "streaming_tmp_dir": ".cache/pioneerml/inference/group_splitter",
        }

    def execute(self, *, model_info: dict, inputs: dict) -> dict:
        cfg = self.get_config()
        threshold = float(cfg.get("threshold", 0.5))
        streaming = self.resolve_streaming_flag(cfg, default=True)
        materialize_outputs = self.resolve_materialize_outputs(cfg, inputs=inputs, default_for_train_mode=True)
        device = self.resolve_device(prefer_cuda=True)

        model_path = model_info["model_path"]
        scripted = self.load_torchscript(model_path=model_path, device=device)

        node_probs_parts: list[torch.Tensor] = []
        node_event_id_parts: list[torch.Tensor] = []
        graph_event_id_parts: list[torch.Tensor] = []
        graph_group_id_parts: list[torch.Tensor] = []

        validated_files = self.resolve_validated_files(inputs)
        flow_cfg = self.resolve_data_flow_config(inputs)
        file_contexts = self.iter_file_contexts(validated_files)

        total_rows = 0
        streamed_prediction_files: list[dict] = []

        streaming_tmp_dir = None
        if streaming:
            tmp_root = str(cfg.get("streaming_tmp_dir", ".cache/pioneerml/inference/group_splitter"))
            streaming_tmp_dir = self.create_streaming_tmp_dir(tmp_root, prefix="run_")

        with torch.no_grad():
            for ctx in file_contexts:
                src_path = ctx.src_path
                n_rows = int(ctx.num_rows)
                total_rows += n_rows

                group_probs_for_file = self.resolve_optional_file_paths(
                    inputs=inputs,
                    file_index=ctx.file_idx,
                    total_files=len(validated_files),
                    source_name="group_probs",
                    input_keys=("group_probs_parquet_paths", "validated_group_probs_files"),
                )
                loader = GroupSplitterGraphLoader(
                    parquet_inputs=self.build_parquet_input_for_file(
                        src_path=src_path,
                        optional_sources={"group_probs": group_probs_for_file},
                    ),
                    mode=str(inputs.get("mode", "inference")),
                    data_flow_config=flow_cfg,
                ).make_dataloader(shuffle_batches=False)

                file_node_probs_parts: list[torch.Tensor] = []
                file_node_event_parts: list[torch.Tensor] = []
                file_graph_event_parts: list[torch.Tensor] = []
                file_graph_group_parts: list[torch.Tensor] = []

                for batch in loader:
                    x = batch.x_node.to(device, non_blocking=(device.type == "cuda"))
                    edge_index = batch.edge_index.to(device, non_blocking=(device.type == "cuda"))
                    edge_attr = batch.x_edge.to(device, non_blocking=(device.type == "cuda"))
                    b = batch.node_graph_id.to(device, non_blocking=(device.type == "cuda"))
                    group_probs = batch.x_graph.to(device, non_blocking=(device.type == "cuda"))

                    logits = scripted(x, edge_index, edge_attr, b, group_probs)
                    if isinstance(logits, (tuple, list)):
                        logits = logits[0]
                    probs = torch.sigmoid(logits).detach().cpu().to(torch.float32)
                    node_event_ids = batch.graph_event_id[batch.node_graph_id].to(torch.int64).cpu()
                    graph_event_id = batch.graph_event_id.to(torch.int64).cpu()
                    graph_time_group_id = batch.graph_time_group_id.to(torch.int64).cpu()

                    file_node_probs_parts.append(probs)
                    file_node_event_parts.append(node_event_ids)
                    file_graph_event_parts.append(graph_event_id)
                    file_graph_group_parts.append(graph_time_group_id)

                    if materialize_outputs:
                        node_probs_parts.append(probs)
                        node_event_id_parts.append(node_event_ids + int(ctx.file_event_offset))
                        graph_event_id_parts.append(graph_event_id + int(ctx.file_event_offset))
                        graph_group_id_parts.append(graph_time_group_id)

                if not file_node_probs_parts:
                    file_node_event_ids_np = torch.empty((0,), dtype=torch.int64).numpy()
                    file_node_probs_np = torch.empty((0, 3), dtype=torch.float32).numpy()
                    file_graph_event_id_np = torch.empty((0,), dtype=torch.int64).numpy()
                    file_graph_time_group_id_np = torch.empty((0,), dtype=torch.int64).numpy()
                else:
                    file_node_event_ids_np = torch.cat(file_node_event_parts, dim=0).numpy().astype("int64", copy=False)
                    file_node_probs_np = torch.cat(file_node_probs_parts, dim=0).numpy().astype("float32", copy=False)
                    file_graph_event_id_np = torch.cat(file_graph_event_parts, dim=0).numpy().astype("int64", copy=False)
                    file_graph_time_group_id_np = torch.cat(file_graph_group_parts, dim=0).numpy().astype("int64", copy=False)

                if streaming:
                    table = BaseTimeGroupOutputAdapterStep.stitch_node_predictions_to_events(
                        node_event_ids_np=file_node_event_ids_np,
                        graph_event_ids_np=file_graph_event_id_np,
                        graph_group_ids_np=file_graph_time_group_id_np,
                        prediction_columns={
                            "pred_hit_pion": file_node_probs_np[:, 0],
                            "pred_hit_muon": file_node_probs_np[:, 1],
                            "pred_hit_mip": file_node_probs_np[:, 2],
                        },
                        num_rows=n_rows,
                    )
                    assert streaming_tmp_dir is not None
                    tmp_pred_path = streaming_tmp_dir / f"{ctx.file_idx:04d}_{src_path.stem}_preds.parquet"
                    self.write_streaming_table(table=table, dst_path=tmp_pred_path)
                    self.append_streamed_file_record(
                        records=streamed_prediction_files,
                        src_path=src_path,
                        prediction_path=tmp_pred_path,
                        num_rows=n_rows,
                    )

        out = {
            "streaming": bool(streaming),
            "streamed_prediction_files": streamed_prediction_files,
            "streaming_tmp_dir": str(streaming_tmp_dir) if streaming_tmp_dir is not None else None,
            "num_rows": int(total_rows),
            "validated_files": validated_files,
            "threshold": float(threshold),
            "model_path": model_path,
        }

        if materialize_outputs and node_probs_parts:
            probs = torch.cat(node_probs_parts, dim=0)
            node_event_ids = torch.cat(node_event_id_parts, dim=0)
            graph_event_id = torch.cat(graph_event_id_parts, dim=0)
            graph_time_group_id = torch.cat(graph_group_id_parts, dim=0)
            out.update(
                {
                    "probs": probs,
                    "node_event_ids": node_event_ids,
                    "graph_event_id": graph_event_id,
                    "graph_time_group_id": graph_time_group_id,
                }
            )
        return out


@step(name="run_group_splitter_inference", enable_cache=False)
def run_group_splitter_inference_step(
    model_info: dict,
    inputs: dict,
    pipeline_config: dict | None = None,
) -> dict:
    return GroupSplitterInferenceRunStep(pipeline_config=pipeline_config).execute(
        model_info=model_info,
        inputs=inputs,
    )
