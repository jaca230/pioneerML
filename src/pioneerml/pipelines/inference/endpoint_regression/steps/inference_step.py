from pathlib import Path

import pyarrow as pa
import torch
from zenml import step

from pioneerml.common.loader import EndpointRegressionGraphLoader
from pioneerml.common.pipeline.steps import BaseInferenceStep, BaseTimeGroupOutputAdapterStep


class EndpointRegressorInferenceRunStep(BaseInferenceStep):
    step_key = "inference"

    _POINT_NAMES = ("start", "end")
    _COORD_NAMES = ("x", "y", "z")
    _QUANTILE_SUFFIXES = ("q16", "q50", "q84")

    def default_config(self) -> dict:
        return {
            "streaming": True,
            "materialize_outputs": None,
            "streaming_tmp_dir": ".cache/pioneerml/inference/endpoint_regressor",
        }

    @classmethod
    def prediction_columns_from_array(cls, preds):
        if preds.ndim != 2:
            raise ValueError(f"Expected predictions to be 2D [N, D], got shape {tuple(preds.shape)}.")
        pred_dim = int(preds.shape[1])
        if pred_dim == 18:
            preds_quant = preds.reshape(-1, 2, 3, 3)
        elif pred_dim == 6:
            preds_quant = preds.reshape(-1, 2, 3, 1).repeat(3, axis=3)
        else:
            raise ValueError(f"Unsupported endpoint prediction dimension {pred_dim}. Expected 6 or 18.")

        out: dict[str, object] = {}
        for point_idx, point_name in enumerate(cls._POINT_NAMES):
            for coord_idx, coord_name in enumerate(cls._COORD_NAMES):
                vals = preds_quant[:, point_idx, coord_idx, :]
                base_name = f"pred_group_{point_name}_{coord_name}"
                out[base_name] = vals[:, 1]
                for q_idx, q_suffix in enumerate(cls._QUANTILE_SUFFIXES):
                    out[f"{base_name}_{q_suffix}"] = vals[:, q_idx]
        return out

    @classmethod
    def prediction_value_types(cls) -> dict[str, pa.DataType]:
        types: dict[str, pa.DataType] = {}
        for point_name in cls._POINT_NAMES:
            for coord_name in cls._COORD_NAMES:
                base_name = f"pred_group_{point_name}_{coord_name}"
                types[base_name] = pa.float32()
                for q_suffix in cls._QUANTILE_SUFFIXES:
                    types[f"{base_name}_{q_suffix}"] = pa.float32()
        return types

    @staticmethod
    def _script_inputs(batch, device: torch.device):
        x = batch.x_node.to(device, non_blocking=(device.type == "cuda"))
        edge_index = batch.edge_index.to(device, non_blocking=(device.type == "cuda"))
        edge_attr = batch.x_edge.to(device, non_blocking=(device.type == "cuda"))
        node_graph_id = batch.node_graph_id.to(device, non_blocking=(device.type == "cuda"))
        x_graph = batch.x_graph.to(device, non_blocking=(device.type == "cuda"))
        group_probs = x_graph[:, :3]
        u = x_graph[:, 3:4] if x_graph.shape[1] >= 4 else x.new_zeros((x_graph.shape[0], 1))
        splitter_probs = x.new_zeros((x.shape[0], 0))
        return x, edge_index, edge_attr, node_graph_id, u, group_probs, splitter_probs

    def execute(self, *, model_info: dict, inputs: dict) -> dict:
        cfg = self.get_config()
        streaming = self.resolve_streaming_flag(cfg, default=True)
        materialize_outputs = self.resolve_materialize_outputs(cfg, inputs=inputs, default_for_train_mode=True)
        if not streaming:
            materialize_outputs = True
        device = self.resolve_device(prefer_cuda=True)

        model_path = model_info["model_path"]
        scripted = self.load_torchscript(model_path=model_path, device=device)

        preds_parts: list[torch.Tensor] = []
        graph_event_id_parts: list[torch.Tensor] = []
        graph_time_group_id_parts: list[torch.Tensor] = []

        validated_files = self.resolve_validated_files(inputs)
        flow_cfg = self.resolve_data_flow_config(inputs)
        file_contexts = self.iter_file_contexts(validated_files)

        total_rows = 0
        streamed_prediction_files: list[dict] = []
        streaming_tmp_dir = None
        prediction_dim: int | None = None

        if streaming:
            tmp_root = str(cfg.get("streaming_tmp_dir", ".cache/pioneerml/inference/endpoint_regressor"))
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
                    input_keys=("validated_group_probs_files", "group_probs_parquet_paths"),
                )
                splitter_for_file = self.resolve_optional_file_paths(
                    inputs=inputs,
                    file_index=ctx.file_idx,
                    total_files=len(validated_files),
                    source_name="group_splitter",
                    input_keys=("validated_group_splitter_files", "group_splitter_parquet_paths"),
                )

                loader = EndpointRegressionGraphLoader(
                    parquet_inputs=self.build_parquet_input_for_file(
                        src_path=src_path,
                        optional_sources={
                            "group_probs": group_probs_for_file,
                            "group_splitter": splitter_for_file,
                        },
                    ),
                    mode=str(inputs.get("mode", "inference")),
                    data_flow_config=flow_cfg,
                ).make_dataloader(shuffle_batches=False)

                file_preds_parts: list[torch.Tensor] = []
                file_event_parts: list[torch.Tensor] = []

                for batch in loader:
                    script_inputs = self._script_inputs(batch, device)
                    preds = scripted(*script_inputs)
                    preds = preds[0] if isinstance(preds, (tuple, list)) else preds
                    preds_cpu = preds.detach().cpu().to(torch.float32)
                    prediction_dim = int(preds_cpu.shape[1]) if preds_cpu.ndim == 2 else prediction_dim

                    event_ids = batch.graph_event_id.to(torch.int64).cpu()
                    file_preds_parts.append(preds_cpu)
                    file_event_parts.append(event_ids)

                    if materialize_outputs:
                        preds_parts.append(preds_cpu)
                        graph_event_id_parts.append(event_ids + int(ctx.file_event_offset))
                        graph_time_group_id_parts.append(batch.graph_time_group_id.to(torch.int64).cpu())

                if not file_preds_parts:
                    file_event_ids_np = torch.empty((0,), dtype=torch.int64).numpy()
                    file_preds_np = torch.empty((0, int(prediction_dim or 0)), dtype=torch.float32).numpy()
                else:
                    file_event_ids_np = torch.cat(file_event_parts, dim=0).numpy().astype("int64", copy=False)
                    file_preds_np = torch.cat(file_preds_parts, dim=0).numpy().astype("float32", copy=False)

                if streaming:
                    table = BaseTimeGroupOutputAdapterStep.stitch_time_group_predictions_to_events(
                        event_ids_np=file_event_ids_np,
                        prediction_columns=self.prediction_columns_from_array(file_preds_np),
                        num_rows=n_rows,
                        value_types=self.prediction_value_types(),
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
            "prediction_dim": int(prediction_dim) if prediction_dim is not None else None,
            "num_rows": int(total_rows),
            "validated_files": validated_files,
            "validated_group_probs_files": [str(p) for p in inputs.get("validated_group_probs_files") or []],
            "validated_group_splitter_files": [str(p) for p in inputs.get("validated_group_splitter_files") or []],
            "model_path": model_path,
        }

        if materialize_outputs and preds_parts:
            out.update(
                {
                    "preds": torch.cat(preds_parts, dim=0),
                    "graph_event_id": torch.cat(graph_event_id_parts, dim=0),
                    "graph_time_group_id": torch.cat(graph_time_group_id_parts, dim=0),
                }
            )
        return out


@step(name="run_endpoint_regressor_inference", enable_cache=False)
def run_endpoint_regressor_inference_step(
    model_info: dict,
    inputs: dict,
    pipeline_config: dict | None = None,
) -> dict:
    return EndpointRegressorInferenceRunStep(pipeline_config=pipeline_config).execute(
        model_info=model_info,
        inputs=inputs,
    )
