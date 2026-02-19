from __future__ import annotations

from pathlib import Path

import torch

from pioneerml.common.loader import GroupSplitterGraphLoader
from pioneerml.common.pipeline.services import BaseInferenceService

from ..base import GroupSplitterInferenceServiceBase


class GroupSplitterInferenceRunService(GroupSplitterInferenceServiceBase, BaseInferenceService):
    step_key = "inference"

    def default_config(self) -> dict:
        return {
            "threshold": 0.5,
            "streaming": True,
            "materialize_outputs": None,
            "streaming_tmp_dir": ".cache/pioneerml/inference/group_splitter",
        }

    def resolve_threshold(self) -> float:
        cfg = self.get_config()
        return float(cfg.get("threshold", 0.5))

    @staticmethod
    def _init_accuracy_counters(num_classes: int = 3) -> dict:
        return {
            "has_targets": False,
            "num_classes": int(num_classes),
            "label_total": 0,
            "label_equal": 0,
            "graph_total": 0,
            "graph_exact": 0,
            "tp": [0 for _ in range(int(num_classes))],
            "fp": [0 for _ in range(int(num_classes))],
            "fn": [0 for _ in range(int(num_classes))],
        }

    @staticmethod
    def _update_accuracy_counters(
        *,
        counters: dict,
        preds_binary: torch.Tensor,
        targets: torch.Tensor,
    ) -> None:
        preds = preds_binary.to(torch.float32)
        truth = targets.to(torch.float32)
        if preds.ndim == 1:
            preds = preds.unsqueeze(1)
        if truth.ndim == 1:
            truth = truth.unsqueeze(1)

        counters["has_targets"] = True
        counters["label_total"] += int(truth.numel())
        counters["label_equal"] += int((preds == truth).sum().item())
        counters["graph_total"] += int(truth.shape[0])
        counters["graph_exact"] += int(((preds == truth).all(dim=1)).sum().item())

        num_classes = int(counters.get("num_classes", truth.shape[1]))
        for cls in range(num_classes):
            cls_truth = truth[:, cls].to(torch.int64)
            cls_pred = preds[:, cls].to(torch.int64)
            counters["fp"][cls] += int(((cls_truth == 0) & (cls_pred == 1)).sum().item())
            counters["fn"][cls] += int(((cls_truth == 1) & (cls_pred == 0)).sum().item())
            counters["tp"][cls] += int(((cls_truth == 1) & (cls_pred == 1)).sum().item())

    def execute(
        self,
        *,
        model_info: dict,
        inputs: dict,
    ) -> dict:
        cfg = self.get_config()
        threshold = self.resolve_threshold()
        streaming = self.resolve_streaming_flag(cfg, default=True)
        materialize_outputs = self.resolve_materialize_outputs(cfg, inputs=inputs, default_for_train_mode=True)
        if not streaming:
            materialize_outputs = True
        device = self.resolve_device(prefer_cuda=True)
        model_path = model_info["model_path"]
        scripted = self.load_torchscript(model_path=model_path, device=device)

        probs_parts: list[torch.Tensor] = []
        target_parts: list[torch.Tensor] = []
        node_event_id_parts: list[torch.Tensor] = []
        node_time_group_parts: list[torch.Tensor] = []

        if "x" in inputs:
            x = inputs["x"].to(device)
            edge_index = inputs["edge_index"].to(device)
            edge_attr = inputs["edge_attr"].to(device)
            batch = inputs["batch"].to(device)
            group_total_energy = inputs["group_total_energy"].to(device)
            group_probs = inputs["group_probs"].to(device)

            with torch.no_grad():
                logits = scripted(x, edge_index, edge_attr, batch, group_total_energy, group_probs)
                if isinstance(logits, (tuple, list)):
                    logits = logits[0]
                probs_parts.append(torch.sigmoid(logits).detach().cpu().to(torch.float32))

            if inputs.get("targets") is not None:
                target_parts.append(inputs["targets"].to(torch.float32))
            node_event_id_parts.append(inputs["node_event_ids"].to(torch.int64))
            node_time_group_parts.append(inputs["node_time_group_ids"].to(torch.int64))
        else:
            validated_files = [str(p) for p in inputs.get("validated_files") or inputs.get("parquet_paths") or []]
            if not validated_files:
                raise RuntimeError("No validated files provided for inference.")

            validated_group_probs = list(inputs.get("validated_group_probs_files") or [])
            counters = self._init_accuracy_counters(num_classes=3)
            total_rows = 0
            streamed_prediction_files: list[dict] = []
            streaming_tmp_dir = None
            file_event_offset = 0

            if streaming:
                tmp_root = str(cfg.get("streaming_tmp_dir", ".cache/pioneerml/inference/group_splitter"))
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

                    loader = GroupSplitterGraphLoader(
                        parquet_paths=[str(src_path)],
                        group_probs_parquet_paths=group_probs_for_file,
                        mode=str(inputs.get("mode", "inference")),
                        use_group_probs=bool(inputs.get("use_group_probs", True)),
                        batch_size=int(inputs["batch_size"]),
                        row_groups_per_chunk=int(inputs["row_groups_per_chunk"]),
                        num_workers=int(inputs["num_workers"]),
                    ).make_dataloader(shuffle_batches=False)

                    file_probs_parts: list[torch.Tensor] = []
                    file_event_parts: list[torch.Tensor] = []
                    file_time_group_parts: list[torch.Tensor] = []

                    for batch in loader:
                        x = batch.x.to(device, non_blocking=(device.type == "cuda"))
                        edge_index = batch.edge_index.to(device, non_blocking=(device.type == "cuda"))
                        edge_attr = batch.edge_attr.to(device, non_blocking=(device.type == "cuda"))
                        b = batch.batch.to(device, non_blocking=(device.type == "cuda"))
                        group_total_energy = batch.group_total_energy.to(device, non_blocking=(device.type == "cuda"))
                        group_probs = batch.group_probs.to(device, non_blocking=(device.type == "cuda"))
                        logits = scripted(x, edge_index, edge_attr, b, group_total_energy, group_probs)
                        if isinstance(logits, (tuple, list)):
                            logits = logits[0]
                        probs = torch.sigmoid(logits).detach().cpu().to(torch.float32)
                        preds_binary = (probs >= threshold).to(torch.float32)

                        local_counts = torch.bincount(
                            batch.batch.to(torch.int64),
                            minlength=int(batch.num_graphs),
                        ).to(torch.int64)
                        node_event_ids = batch.event_ids.to(torch.int64).repeat_interleave(local_counts).cpu()
                        node_time_group_ids = batch.time_group_ids.to(torch.int64).cpu()
                        file_probs_parts.append(probs)
                        file_event_parts.append(node_event_ids)
                        file_time_group_parts.append(node_time_group_ids)

                        if materialize_outputs:
                            probs_parts.append(probs)
                            node_event_id_parts.append(node_event_ids + int(file_event_offset))
                            node_time_group_parts.append(node_time_group_ids)

                        if hasattr(batch, "y") and batch.y is not None:
                            truth = batch.y.detach().cpu().to(torch.float32)
                            self._update_accuracy_counters(counters=counters, preds_binary=preds_binary, targets=truth)
                            if materialize_outputs:
                                target_parts.append(truth)

                    if not file_probs_parts:
                        file_event_ids_np = torch.empty((0,), dtype=torch.int64).numpy()
                        file_time_group_np = torch.empty((0,), dtype=torch.int64).numpy()
                        file_probs_np = torch.empty((0, 3), dtype=torch.float32).numpy()
                    else:
                        file_event_ids_np = torch.cat(file_event_parts, dim=0).numpy().astype("int64", copy=False)
                        file_time_group_np = torch.cat(file_time_group_parts, dim=0).numpy().astype("int64", copy=False)
                        file_probs_np = torch.cat(file_probs_parts, dim=0).numpy().astype("float32", copy=False)

                    if streaming:
                        table = self.build_prediction_table(
                            event_ids_np=file_event_ids_np,
                            time_group_ids_np=file_time_group_np,
                            probs_np=file_probs_np,
                            num_rows=n_rows,
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
                    "accuracy_counters": counters,
                    "num_rows": int(total_rows),
                    "validated_files": validated_files,
                    "validated_group_probs_files": validated_group_probs,
                    "threshold": float(threshold),
                    "model_path": model_path,
                }
                if materialize_outputs:
                    if not probs_parts:
                        raise RuntimeError("No inference outputs produced from inputs.")
                    probs = torch.cat(probs_parts, dim=0)
                    targets = torch.cat(target_parts, dim=0) if target_parts else None
                    node_event_ids = torch.cat(node_event_id_parts, dim=0)
                    node_time_group_ids = torch.cat(node_time_group_parts, dim=0)
                    preds_binary = (probs >= threshold).to(torch.float32)
                    out.update(
                        {
                            "probs": probs,
                            "preds_binary": preds_binary,
                            "targets": targets,
                            "node_event_ids": node_event_ids,
                            "node_time_group_ids": node_time_group_ids,
                        }
                    )
                return out

        if not probs_parts:
            raise RuntimeError("No inference outputs produced from inputs.")

        probs = torch.cat(probs_parts, dim=0)
        targets = torch.cat(target_parts, dim=0) if target_parts else None
        node_event_ids = torch.cat(node_event_id_parts, dim=0)
        node_time_group_ids = torch.cat(node_time_group_parts, dim=0)
        preds_binary = (probs >= threshold).to(torch.float32)

        return {
            "probs": probs,
            "preds_binary": preds_binary,
            "targets": targets,
            "node_event_ids": node_event_ids,
            "node_time_group_ids": node_time_group_ids,
            "num_rows": int(inputs.get("num_rows", 0)),
            "validated_files": list(inputs.get("validated_files") or []),
            "validated_group_probs_files": list(inputs.get("validated_group_probs_files") or []),
            "threshold": float(threshold),
            "model_path": model_path,
        }
