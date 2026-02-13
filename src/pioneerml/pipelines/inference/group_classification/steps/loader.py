from pathlib import Path

import pyarrow.parquet as pq
import torch
from zenml import step

from pioneerml.common.loader import GroupClassifierGraphLoader


def _resolve_config(pipeline_config: dict | None) -> dict:
    if pipeline_config is None:
        return {}
    if not isinstance(pipeline_config, dict):
        raise TypeError(f"Expected dict pipeline_config, got {type(pipeline_config).__name__}.")
    raw = pipeline_config.get("loader")
    if raw is None:
        return {}
    if not isinstance(raw, dict):
        raise TypeError(f"Expected dict for loader config, got {type(raw).__name__}.")
    return dict(raw)


def _resolve_inference_runtime(config_json: dict) -> tuple[str, int, int, int]:
    mode = str(config_json.get("mode", "inference")).strip().lower()
    if mode not in {"inference", "train"}:
        raise ValueError(f"Unsupported loader mode: {mode}. Expected 'inference' or 'train'.")
    batch_size = max(1, int(config_json.get("batch_size", 64)))
    row_groups_per_chunk = max(1, int(config_json.get("chunk_row_groups", config_json.get("row_groups_per_chunk", 4))))
    num_workers = max(0, int(config_json.get("chunk_workers", config_json.get("num_workers", 0))))
    return mode, batch_size, row_groups_per_chunk, num_workers


def _count_input_rows(parquet_paths: list[str]) -> int:
    total = 0
    for p in parquet_paths:
        total += int(pq.ParquetFile(p).metadata.num_rows)
    return total


def load_group_classifier_inference_inputs_local(
    parquet_paths: list[str],
    *,
    config_json: dict | None = None,
    normalize: bool = False,
    normalize_eps: float = 1e-6,
) -> dict:
    _ = normalize
    _ = normalize_eps
    cfg = dict(config_json or {})
    mode, batch_size, row_groups_per_chunk, num_workers = _resolve_inference_runtime(cfg)
    loader = GroupClassifierGraphLoader(
        parquet_paths=[str(Path(p).expanduser().resolve()) for p in parquet_paths],
        mode=mode,
        batch_size=batch_size,
        row_groups_per_chunk=row_groups_per_chunk,
        num_workers=num_workers,
    )
    dl = loader.make_dataloader(shuffle_batches=False)

    x_parts: list[torch.Tensor] = []
    edge_index_parts: list[torch.Tensor] = []
    edge_attr_parts: list[torch.Tensor] = []
    batch_parts: list[torch.Tensor] = []
    graph_event_id_parts: list[torch.Tensor] = []
    graph_group_id_parts: list[torch.Tensor] = []

    node_offset = 0
    graph_offset = 0

    for batch in dl:
        x = batch.x.to(torch.float32).cpu()
        edge_index = batch.edge_index.to(torch.int64).cpu()
        edge_attr = batch.edge_attr.to(torch.float32).cpu()
        b = batch.batch.to(torch.int64).cpu()
        event_ids = batch.event_ids.to(torch.int64).cpu()
        group_ids = batch.group_ids.to(torch.int64).cpu()

        if x.numel() > 0:
            x_parts.append(x)
        if edge_index.numel() > 0:
            edge_index_parts.append(edge_index + int(node_offset))
            edge_attr_parts.append(edge_attr)
        if b.numel() > 0:
            batch_parts.append(b + int(graph_offset))

        graph_event_id_parts.append(event_ids)
        graph_group_id_parts.append(group_ids)

        node_offset += int(x.shape[0])
        graph_offset += int(batch.num_graphs)

    if graph_offset <= 0:
        raise RuntimeError("No group graphs created from input data.")

    x_out = torch.cat(x_parts, dim=0) if x_parts else torch.empty((0, 4), dtype=torch.float32)
    edge_index_out = (
        torch.cat(edge_index_parts, dim=1) if edge_index_parts else torch.empty((2, 0), dtype=torch.int64)
    )
    edge_attr_out = torch.cat(edge_attr_parts, dim=0) if edge_attr_parts else torch.empty((0, 4), dtype=torch.float32)
    batch_out = torch.cat(batch_parts, dim=0) if batch_parts else torch.empty((0,), dtype=torch.int64)
    graph_event_ids = torch.cat(graph_event_id_parts, dim=0) if graph_event_id_parts else torch.empty((0,), dtype=torch.int64)
    graph_group_ids = torch.cat(graph_group_id_parts, dim=0) if graph_group_id_parts else torch.empty((0,), dtype=torch.int64)
    targets_out = torch.zeros((graph_offset, 3), dtype=torch.float32)

    return {
        "x": x_out,
        "edge_index": edge_index_out,
        "edge_attr": edge_attr_out,
        "batch": batch_out,
        "targets": targets_out,
        "num_rows": _count_input_rows(loader.parquet_paths),
        "graph_event_ids": graph_event_ids,
        "graph_group_ids": graph_group_ids,
        "validated_files": list(loader.parquet_paths),
    }


@step(enable_cache=False)
def load_group_classifier_inference_inputs(
    parquet_paths: list[str],
    pipeline_config: dict | None = None,
) -> dict:
    cfg = _resolve_config(pipeline_config)
    config_json = dict(cfg.get("config_json") or {})
    mode, batch_size, row_groups_per_chunk, num_workers = _resolve_inference_runtime(config_json)
    resolved = [str(Path(p).expanduser().resolve()) for p in parquet_paths]
    if not resolved:
        raise RuntimeError("No parquet paths provided for inference.")

    return {
        "input_mode": "graph_loader_inference_v1",
        "mode": mode,
        "parquet_paths": resolved,
        "batch_size": int(batch_size),
        "row_groups_per_chunk": int(row_groups_per_chunk),
        "num_workers": int(num_workers),
        "num_rows": int(_count_input_rows(resolved)),
        "validated_files": list(resolved),
    }
