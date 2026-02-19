from __future__ import annotations

from pathlib import Path

import torch

from pioneerml.common.loader import PositronAngleGraphLoader
from pioneerml.common.pipeline.services import BaseInferenceService

from ..base import PositronAngleInferenceServiceBase


class PositronAngleInferenceRunService(PositronAngleInferenceServiceBase, BaseInferenceService):
    step_key = "inference"

    def default_config(self) -> dict:
        return {
            "streaming": True,
            "materialize_outputs": None,
            "streaming_tmp_dir": ".cache/pioneerml/inference/positron_angle",
            "prediction_cuts": {
                "enabled": True,
                "group_prob_threshold": 0.5,
            },
        }

    @staticmethod
    def _init_regression_counters() -> dict:
        return {"has_targets": False, "sum_sq_error": 0.0, "sum_abs_error": 0.0, "count": 0}

    @staticmethod
    def _update_regression_counters(*, counters: dict, preds: torch.Tensor, targets: torch.Tensor) -> None:
        p = preds.to(torch.float32)
        t = targets.to(torch.float32)
        err = p - t
        counters["has_targets"] = True
        counters["sum_sq_error"] += float((err * err).sum().item())
        counters["sum_abs_error"] += float(torch.abs(err).sum().item())
        counters["count"] += int(err.numel())

    @staticmethod
    def _compute_prediction_mask(
        *,
        group_probs: torch.Tensor,
        group_prob_threshold: float,
    ) -> torch.Tensor:
        if group_probs.ndim != 2 or group_probs.shape[1] < 3:
            return torch.ones((group_probs.shape[0],), dtype=torch.bool)
        g = group_probs.to(torch.float32)
        is_pion = g[:, 0] > float(group_prob_threshold)
        is_mip = g[:, 2] > float(group_prob_threshold)
        # Match Omar's precedence: if both pion and mip pass threshold, treat as pion-side.
        return is_mip & (~is_pion)

    def execute(
        self,
        *,
        model_info: dict,
        inputs: dict,
    ) -> dict:
        cfg = self.get_config()
        streaming = self.resolve_streaming_flag(cfg, default=True)
        materialize_outputs = self.resolve_materialize_outputs(cfg, inputs=inputs, default_for_train_mode=True)
        if not streaming:
            materialize_outputs = True
        cuts_cfg = dict(cfg.get("prediction_cuts") or {})
        use_prediction_cuts = bool(cuts_cfg.get("enabled", True))
        group_prob_threshold = float(cuts_cfg.get("group_prob_threshold", 0.5))
        device = self.resolve_device(prefer_cuda=True)
        model_path = model_info["model_path"]
        scripted = self.load_torchscript(model_path=model_path, device=device)

        preds_parts: list[torch.Tensor] = []
        target_parts: list[torch.Tensor] = []
        graph_event_id_parts: list[torch.Tensor] = []
        graph_group_id_parts: list[torch.Tensor] = []

        validated_files = [str(p) for p in inputs.get("validated_files") or inputs.get("parquet_paths") or []]
        if not validated_files:
            raise RuntimeError("No validated files provided for inference.")
        validated_group_probs = list(inputs.get("validated_group_probs_files") or [])
        validated_group_splitter = list(inputs.get("validated_group_splitter_files") or [])
        validated_endpoint = list(inputs.get("validated_endpoint_files") or [])
        validated_event_splitter = list(inputs.get("validated_event_splitter_files") or [])
        validated_pion_stop = list(inputs.get("validated_pion_stop_files") or [])
        counters = self._init_regression_counters()
        total_rows = 0
        streamed_prediction_files: list[dict] = []
        streaming_tmp_dir = None
        file_event_offset = 0
        prediction_dim: int | None = None

        if streaming:
            tmp_root = str(cfg.get("streaming_tmp_dir", ".cache/pioneerml/inference/positron_angle"))
            streaming_tmp_dir = self.create_streaming_tmp_dir(tmp_root, prefix="run_")

        with torch.no_grad():
            for file_idx, src_file in enumerate(validated_files):
                src_path = Path(src_file).expanduser().resolve()
                n_rows = int(self.count_input_rows([str(src_path)]))
                total_rows += n_rows
                group_probs_for_file = self.select_file_aligned_paths(
                    validated_group_probs,
                    file_index=file_idx,
                    total_files=len(validated_files),
                    label="group_probs",
                )
                splitter_for_file = self.select_file_aligned_paths(
                    validated_group_splitter,
                    file_index=file_idx,
                    total_files=len(validated_files),
                    label="group_splitter",
                )
                endpoint_for_file = self.select_file_aligned_paths(
                    validated_endpoint,
                    file_index=file_idx,
                    total_files=len(validated_files),
                    label="endpoint",
                )
                event_splitter_for_file = self.select_file_aligned_paths(
                    validated_event_splitter,
                    file_index=file_idx,
                    total_files=len(validated_files),
                    label="event_splitter",
                )
                pion_stop_for_file = self.select_file_aligned_paths(
                    validated_pion_stop,
                    file_index=file_idx,
                    total_files=len(validated_files),
                    label="pion_stop",
                )
                loader = PositronAngleGraphLoader(
                    parquet_paths=[str(src_path)],
                    group_probs_parquet_paths=group_probs_for_file,
                    group_splitter_parquet_paths=splitter_for_file,
                    endpoint_parquet_paths=endpoint_for_file,
                    event_splitter_parquet_paths=event_splitter_for_file,
                    pion_stop_parquet_paths=pion_stop_for_file,
                    mode=str(inputs.get("mode", "inference")),
                    use_group_probs=bool(inputs.get("use_group_probs", True)),
                    use_splitter_probs=bool(inputs.get("use_splitter_probs", True)),
                    use_endpoint_preds=bool(inputs.get("use_endpoint_preds", True)),
                    use_event_splitter_affinity=bool(inputs.get("use_event_splitter_affinity", True)),
                    use_pion_stop_preds=bool(inputs.get("use_pion_stop_preds", True)),
                    batch_size=int(inputs["batch_size"]),
                    row_groups_per_chunk=int(inputs["row_groups_per_chunk"]),
                    num_workers=int(inputs["num_workers"]),
                ).make_dataloader(shuffle_batches=False)

                file_preds_parts: list[torch.Tensor] = []
                file_event_parts: list[torch.Tensor] = []
                file_group_parts: list[torch.Tensor] = []

                for batch in loader:
                    x = batch.x.to(device, non_blocking=(device.type == "cuda"))
                    edge_index = batch.edge_index.to(device, non_blocking=(device.type == "cuda"))
                    edge_attr = batch.edge_attr.to(device, non_blocking=(device.type == "cuda"))
                    b = batch.batch.to(device, non_blocking=(device.type == "cuda"))
                    group_probs = batch.group_probs.to(device, non_blocking=(device.type == "cuda"))
                    splitter_probs = batch.splitter_probs.to(device, non_blocking=(device.type == "cuda"))
                    endpoint_preds = batch.endpoint_preds.to(device, non_blocking=(device.type == "cuda"))
                    event_affinity = batch.event_affinity.to(device, non_blocking=(device.type == "cuda"))
                    pion_stop_preds = batch.pion_stop_preds.to(device, non_blocking=(device.type == "cuda"))
                    preds = scripted(
                        x,
                        edge_index,
                        edge_attr,
                        b,
                        group_probs,
                        splitter_probs,
                        endpoint_preds,
                        event_affinity,
                        pion_stop_preds,
                    )
                    preds = preds[0] if isinstance(preds, (tuple, list)) else preds
                    preds_cpu = preds.detach().cpu().to(torch.float32)
                    prediction_dim = int(preds_cpu.shape[1]) if preds_cpu.ndim == 2 else prediction_dim

                    event_ids = batch.event_ids.to(torch.int64).cpu()
                    group_ids = batch.group_ids.to(torch.int64).cpu()
                    if use_prediction_cuts:
                        keep_mask = self._compute_prediction_mask(
                            group_probs=batch.group_probs.to(torch.float32).cpu(),
                            group_prob_threshold=group_prob_threshold,
                        )
                        if keep_mask.shape[0] != preds_cpu.shape[0]:
                            raise RuntimeError(
                                "Prediction-cut mask size mismatch in positron-angle inference: "
                                f"mask={keep_mask.shape[0]} preds={preds_cpu.shape[0]}"
                            )
                        if not torch.any(keep_mask):
                            continue
                        preds_cpu = preds_cpu[keep_mask]
                        event_ids = event_ids[keep_mask]
                        group_ids = group_ids[keep_mask]
                    file_preds_parts.append(preds_cpu)
                    file_event_parts.append(event_ids)
                    file_group_parts.append(group_ids)

                    if materialize_outputs:
                        preds_parts.append(preds_cpu)
                        graph_event_id_parts.append(event_ids + int(file_event_offset))
                        graph_group_id_parts.append(group_ids)

                    if hasattr(batch, "y") and batch.y is not None:
                        truth = batch.y.detach().cpu().to(torch.float32)
                        if use_prediction_cuts:
                            truth = truth[keep_mask]
                        self._update_regression_counters(counters=counters, preds=preds_cpu, targets=truth)
                        if materialize_outputs:
                            target_parts.append(truth)

                if not file_preds_parts:
                    file_event_ids_np = torch.empty((0,), dtype=torch.int64).numpy()
                    file_group_ids_np = torch.empty((0,), dtype=torch.int64).numpy()
                    file_preds_np = torch.empty((0, int(prediction_dim or 0)), dtype=torch.float32).numpy()
                else:
                    file_event_ids_np = torch.cat(file_event_parts, dim=0).numpy().astype("int64", copy=False)
                    file_group_ids_np = torch.cat(file_group_parts, dim=0).numpy().astype("int64", copy=False)
                    file_preds_np = torch.cat(file_preds_parts, dim=0).numpy().astype("float32", copy=False)

                if streaming:
                    table = self.build_prediction_table(
                        event_ids_np=file_event_ids_np,
                        group_ids_np=file_group_ids_np,
                        preds_np=file_preds_np,
                        num_rows=n_rows,
                        source_parquet_path=str(src_path),
                    )
                    assert streaming_tmp_dir is not None
                    tmp_pred_path = streaming_tmp_dir / f"{file_idx:04d}_{src_path.stem}_preds.parquet"
                    self.write_streaming_table(table=table, dst_path=tmp_pred_path)
                    streamed_prediction_files.append(
                        {
                            "source_path": str(src_path),
                            "prediction_path": str(tmp_pred_path),
                            "num_rows": int(n_rows),
                        }
                    )

                file_event_offset += n_rows

        if streaming:
            out = {
                "streaming": True,
                "streamed_prediction_files": streamed_prediction_files,
                "streaming_tmp_dir": str(streaming_tmp_dir) if streaming_tmp_dir is not None else None,
                "regression_counters": counters,
                "prediction_dim": int(prediction_dim) if prediction_dim is not None else None,
                "num_rows": int(total_rows),
                "validated_files": validated_files,
                "validated_group_probs_files": validated_group_probs,
                "validated_group_splitter_files": validated_group_splitter,
                "validated_endpoint_files": validated_endpoint,
                "validated_event_splitter_files": validated_event_splitter,
                "validated_pion_stop_files": validated_pion_stop,
                "model_path": model_path,
            }
            if materialize_outputs:
                if not preds_parts:
                    raise RuntimeError("No inference outputs produced from inputs.")
                preds = torch.cat(preds_parts, dim=0)
                targets = torch.cat(target_parts, dim=0) if target_parts else None
                graph_event_ids = torch.cat(graph_event_id_parts, dim=0)
                graph_group_ids = torch.cat(graph_group_id_parts, dim=0)
                out.update(
                    {
                        "preds": preds,
                        "targets": targets,
                        "graph_event_ids": graph_event_ids,
                        "graph_group_ids": graph_group_ids,
                    }
                )
            return out

        if not preds_parts:
            raise RuntimeError("No inference outputs produced from inputs.")

        preds = torch.cat(preds_parts, dim=0)
        targets = torch.cat(target_parts, dim=0) if target_parts else None
        graph_event_ids = torch.cat(graph_event_id_parts, dim=0)
        graph_group_ids = torch.cat(graph_group_id_parts, dim=0)

        return {
            "preds": preds,
            "targets": targets,
            "graph_event_ids": graph_event_ids,
            "graph_group_ids": graph_group_ids,
            "num_rows": int(inputs.get("num_rows", 0)),
            "validated_files": list(inputs.get("validated_files") or []),
            "validated_group_probs_files": list(inputs.get("validated_group_probs_files") or []),
            "validated_group_splitter_files": list(inputs.get("validated_group_splitter_files") or []),
            "validated_endpoint_files": list(inputs.get("validated_endpoint_files") or []),
            "validated_event_splitter_files": list(inputs.get("validated_event_splitter_files") or []),
            "validated_pion_stop_files": list(inputs.get("validated_pion_stop_files") or []),
            "model_path": model_path,
        }
