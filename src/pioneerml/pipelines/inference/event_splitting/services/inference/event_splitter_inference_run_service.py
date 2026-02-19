from __future__ import annotations

from pathlib import Path

import torch

from pioneerml.common.loader import EventSplitterGraphLoader
from pioneerml.common.pipeline.services import BaseInferenceService

from ..base import EventSplitterInferenceServiceBase


class EventSplitterInferenceRunService(EventSplitterInferenceServiceBase, BaseInferenceService):
    step_key = "inference"

    def default_config(self) -> dict:
        return {
            "threshold": 0.5,
            "streaming": True,
            "materialize_outputs": None,
            "streaming_tmp_dir": ".cache/pioneerml/inference/event_splitter",
            "debug_memory": False,
            "debug_memory_every_batches": 10,
        }

    def resolve_threshold(self) -> float:
        cfg = self.get_config()
        return float(cfg.get("threshold", 0.5))

    @staticmethod
    def _init_accuracy_counters() -> dict:
        return {"has_targets": False, "label_total": 0, "label_equal": 0, "tp": 0, "fp": 0, "fn": 0}

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
        truth_i = truth.to(torch.int64)
        pred_i = preds.to(torch.int64)
        counters["fp"] += int(((truth_i == 0) & (pred_i == 1)).sum().item())
        counters["fn"] += int(((truth_i == 1) & (pred_i == 0)).sum().item())
        counters["tp"] += int(((truth_i == 1) & (pred_i == 1)).sum().item())

    def execute(
        self,
        *,
        model_info: dict,
        inputs: dict,
    ) -> dict:
        cfg = self.get_config()
        threshold = float(cfg.get("threshold", 0.5))
        streaming = self.resolve_streaming_flag(cfg, default=True)
        materialize_outputs = self.resolve_materialize_outputs(cfg, inputs=inputs, default_for_train_mode=True)
        if not streaming:
            materialize_outputs = True
        debug_memory = bool(cfg.get("debug_memory", False))
        debug_every = max(1, int(cfg.get("debug_memory_every_batches", 10)))

        def _log_mem(tag: str) -> None:
            if not debug_memory:
                return
            try:
                import os
                import psutil

                rss_gib = psutil.Process(os.getpid()).memory_info().rss / (1024**3)
                print(f"[run_event_splitter_inference][mem] {tag}: rss={rss_gib:.2f} GiB")
            except Exception:
                pass

        device = self.resolve_device(prefer_cuda=True)
        model_path = model_info["model_path"]
        scripted = self.load_torchscript(model_path=model_path, device=device)
        _log_mem("after model load")

        validated_files = [str(p) for p in inputs.get("validated_files") or inputs.get("parquet_paths") or []]
        if not validated_files:
            raise RuntimeError("No validated files provided for inference.")
        validated_group_probs = list(inputs.get("validated_group_probs_files") or [])
        validated_group_splitter = list(inputs.get("validated_group_splitter_files") or [])
        validated_endpoint = list(inputs.get("validated_endpoint_files") or [])

        prob_parts: list[torch.Tensor] = []
        logit_parts: list[torch.Tensor] = []
        target_parts: list[torch.Tensor] = []

        edge_event_id_parts: list[torch.Tensor] = []
        edge_src_local_parts: list[torch.Tensor] = []
        edge_dst_local_parts: list[torch.Tensor] = []
        node_event_id_parts: list[torch.Tensor] = []
        node_local_idx_parts: list[torch.Tensor] = []
        node_time_group_parts: list[torch.Tensor] = []

        counters = self._init_accuracy_counters()
        total_rows = 0
        logit_sum = 0.0
        logit_count = 0
        streamed_prediction_files: list[dict] = []
        streaming_tmp_dir = None
        file_event_offset = 0
        if streaming:
            tmp_root = str(cfg.get("streaming_tmp_dir", ".cache/pioneerml/inference/event_splitter"))
            streaming_tmp_dir = self.create_streaming_tmp_dir(tmp_root, prefix="run_")

        _log_mem("after loader init")
        with torch.no_grad():
            batch_idx = 0
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

                loader = EventSplitterGraphLoader(
                    parquet_paths=[str(src_path)],
                    group_probs_parquet_paths=group_probs_for_file,
                    group_splitter_parquet_paths=splitter_for_file,
                    endpoint_parquet_paths=endpoint_for_file,
                    mode=str(inputs.get("mode", "inference")),
                    use_group_probs=bool(inputs.get("use_group_probs", True)),
                    use_splitter_probs=bool(inputs.get("use_splitter_probs", True)),
                    use_endpoint_preds=bool(inputs.get("use_endpoint_preds", True)),
                    batch_size=int(inputs["batch_size"]),
                    row_groups_per_chunk=int(inputs["row_groups_per_chunk"]),
                    num_workers=int(inputs["num_workers"]),
                ).make_dataloader(shuffle_batches=False)

                file_prob_parts: list[torch.Tensor] = []
                file_edge_event_parts: list[torch.Tensor] = []
                file_edge_src_parts: list[torch.Tensor] = []
                file_edge_dst_parts: list[torch.Tensor] = []
                file_node_event_parts: list[torch.Tensor] = []
                file_node_idx_parts: list[torch.Tensor] = []
                file_node_tg_parts: list[torch.Tensor] = []

                for batch in loader:
                    batch_idx += 1
                    cpu_batch = batch.batch.to(torch.int64).cpu()
                    cpu_edge_index = batch.edge_index.to(torch.int64).cpu()
                    cpu_event_ids = batch.event_ids.to(torch.int64).cpu()
                    cpu_time_groups = batch.time_group_ids.to(torch.int64).cpu()

                    local_counts = torch.bincount(cpu_batch, minlength=int(batch.num_graphs)).to(torch.int64)
                    node_ptr = torch.zeros((int(batch.num_graphs) + 1,), dtype=torch.int64)
                    node_ptr[1:] = torch.cumsum(local_counts, dim=0)

                    edge_src = cpu_edge_index[0]
                    edge_dst = cpu_edge_index[1]
                    edge_event_local = cpu_batch[edge_src]
                    edge_event_ids = cpu_event_ids[edge_event_local]
                    edge_src_local = edge_src - node_ptr[edge_event_local]
                    edge_dst_local = edge_dst - node_ptr[edge_event_local]

                    node_event_ids = cpu_event_ids.repeat_interleave(local_counts)
                    node_local_idx = torch.arange(cpu_batch.numel(), dtype=torch.int64) - node_ptr[cpu_batch]

                    x = batch.x.to(device, non_blocking=(device.type == "cuda"))
                    edge_index = batch.edge_index.to(device, non_blocking=(device.type == "cuda"))
                    edge_attr = batch.edge_attr.to(device, non_blocking=(device.type == "cuda"))
                    b = batch.batch.to(device, non_blocking=(device.type == "cuda"))
                    group_ptr = batch.group_ptr.to(device, non_blocking=(device.type == "cuda"))
                    time_group_ids = batch.time_group_ids.to(device, non_blocking=(device.type == "cuda"))
                    group_probs = batch.group_probs.to(device, non_blocking=(device.type == "cuda"))
                    splitter_probs = batch.splitter_probs.to(device, non_blocking=(device.type == "cuda"))
                    endpoint_preds = batch.endpoint_preds.to(device, non_blocking=(device.type == "cuda"))

                    logits = scripted(
                        x,
                        edge_index,
                        edge_attr,
                        b,
                        group_ptr,
                        time_group_ids,
                        group_probs,
                        splitter_probs,
                        endpoint_preds,
                    )
                    if isinstance(logits, (tuple, list)):
                        logits = logits[0]
                    probs = torch.sigmoid(logits)

                    logits_cpu = logits.detach().cpu().to(torch.float32)
                    probs_cpu = probs.detach().cpu().to(torch.float32)
                    preds_binary = (probs_cpu >= threshold).to(torch.float32)
                    logit_sum += float(logits_cpu.sum().item())
                    logit_count += int(logits_cpu.numel())

                    file_prob_parts.append(probs_cpu)
                    file_edge_event_parts.append(edge_event_ids)
                    file_edge_src_parts.append(edge_src_local)
                    file_edge_dst_parts.append(edge_dst_local)
                    file_node_event_parts.append(node_event_ids)
                    file_node_idx_parts.append(node_local_idx)
                    file_node_tg_parts.append(cpu_time_groups)

                    if materialize_outputs:
                        prob_parts.append(probs_cpu)
                        logit_parts.append(logits_cpu)
                        edge_event_id_parts.append(edge_event_ids + int(file_event_offset))
                        edge_src_local_parts.append(edge_src_local)
                        edge_dst_local_parts.append(edge_dst_local)
                        node_event_id_parts.append(node_event_ids + int(file_event_offset))
                        node_local_idx_parts.append(node_local_idx)
                        node_time_group_parts.append(cpu_time_groups)

                    if hasattr(batch, "y") and batch.y is not None:
                        truth = batch.y.detach().cpu().to(torch.float32)
                        self._update_accuracy_counters(counters=counters, preds_binary=preds_binary, targets=truth)
                        if materialize_outputs:
                            target_parts.append(truth)

                    if batch_idx % debug_every == 0:
                        _log_mem(f"after batch {batch_idx}")

                if not file_prob_parts:
                    file_edge_probs_np = torch.empty((0,), dtype=torch.float32).numpy()
                    file_edge_event_ids_np = torch.empty((0,), dtype=torch.int64).numpy()
                    file_edge_src_local_np = torch.empty((0,), dtype=torch.int64).numpy()
                    file_edge_dst_local_np = torch.empty((0,), dtype=torch.int64).numpy()
                    file_node_event_ids_np = torch.empty((0,), dtype=torch.int64).numpy()
                    file_node_local_idx_np = torch.empty((0,), dtype=torch.int64).numpy()
                    file_node_tg_np = torch.empty((0,), dtype=torch.int64).numpy()
                else:
                    file_edge_probs_np = (
                        torch.cat(file_prob_parts, dim=0).reshape(-1).numpy().astype("float32", copy=False)
                    )
                    file_edge_event_ids_np = (
                        torch.cat(file_edge_event_parts, dim=0).numpy().astype("int64", copy=False)
                    )
                    file_edge_src_local_np = (
                        torch.cat(file_edge_src_parts, dim=0).numpy().astype("int64", copy=False)
                    )
                    file_edge_dst_local_np = (
                        torch.cat(file_edge_dst_parts, dim=0).numpy().astype("int64", copy=False)
                    )
                    file_node_event_ids_np = (
                        torch.cat(file_node_event_parts, dim=0).numpy().astype("int64", copy=False)
                    )
                    file_node_local_idx_np = (
                        torch.cat(file_node_idx_parts, dim=0).numpy().astype("int64", copy=False)
                    )
                    file_node_tg_np = torch.cat(file_node_tg_parts, dim=0).numpy().astype("int64", copy=False)

                if streaming:
                    table = self.build_prediction_table(
                        node_event_ids_np=file_node_event_ids_np,
                        node_local_idx_np=file_node_local_idx_np,
                        node_time_group_ids_np=file_node_tg_np,
                        edge_event_ids_np=file_edge_event_ids_np,
                        edge_src_local_np=file_edge_src_local_np,
                        edge_dst_local_np=file_edge_dst_local_np,
                        edge_probs_np=file_edge_probs_np,
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

        if materialize_outputs and not prob_parts:
            raise RuntimeError("No inference outputs produced from inputs.")

        _log_mem("before final cat")
        mean_logit = (logit_sum / float(logit_count)) if logit_count > 0 else None

        if streaming:
            out = {
                "streaming": True,
                "streamed_prediction_files": streamed_prediction_files,
                "streaming_tmp_dir": str(streaming_tmp_dir) if streaming_tmp_dir is not None else None,
                "accuracy_counters": counters,
                "mean_logit": mean_logit,
                "num_rows": int(total_rows),
                "validated_files": validated_files,
                "validated_group_probs_files": validated_group_probs,
                "validated_group_splitter_files": validated_group_splitter,
                "validated_endpoint_files": validated_endpoint,
                "threshold": float(threshold),
                "model_path": model_path,
            }
            if materialize_outputs:
                logits = torch.cat(logit_parts, dim=0)
                probs = torch.cat(prob_parts, dim=0)
                preds_binary = (probs >= threshold).to(torch.float32)
                targets = torch.cat(target_parts, dim=0) if target_parts else None
                out.update(
                    {
                        "logits": logits,
                        "probs": probs,
                        "preds_binary": preds_binary,
                        "targets": targets,
                        "edge_event_ids": torch.cat(edge_event_id_parts, dim=0),
                        "edge_src_local": torch.cat(edge_src_local_parts, dim=0),
                        "edge_dst_local": torch.cat(edge_dst_local_parts, dim=0),
                        "node_event_ids": torch.cat(node_event_id_parts, dim=0),
                        "node_local_index": torch.cat(node_local_idx_parts, dim=0),
                        "node_time_group_ids": torch.cat(node_time_group_parts, dim=0),
                    }
                )
            _log_mem("before return")
            return out

        logits = torch.cat(logit_parts, dim=0)
        probs = torch.cat(prob_parts, dim=0)
        preds_binary = (probs >= threshold).to(torch.float32)
        targets = torch.cat(target_parts, dim=0) if target_parts else None
        _log_mem("after final cat")

        out = {
            "logits": logits,
            "probs": probs,
            "preds_binary": preds_binary,
            "targets": targets,
            "edge_event_ids": torch.cat(edge_event_id_parts, dim=0),
            "edge_src_local": torch.cat(edge_src_local_parts, dim=0),
            "edge_dst_local": torch.cat(edge_dst_local_parts, dim=0),
            "node_event_ids": torch.cat(node_event_id_parts, dim=0),
            "node_local_index": torch.cat(node_local_idx_parts, dim=0),
            "node_time_group_ids": torch.cat(node_time_group_parts, dim=0),
            "accuracy_counters": counters,
            "mean_logit": mean_logit,
            "num_rows": int(total_rows),
            "validated_files": validated_files,
            "validated_group_probs_files": validated_group_probs,
            "validated_group_splitter_files": validated_group_splitter,
            "validated_endpoint_files": validated_endpoint,
            "threshold": float(threshold),
            "model_path": model_path,
        }
        _log_mem("before return")
        return out
