from pathlib import Path

import pyarrow.parquet as pq
import torch
from zenml import step

from pioneerml.common.loader import GroupSplitterGraphLoader


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


def _resolve_inference_runtime(config_json: dict) -> tuple[str, bool, int, int, int]:
    mode = str(config_json.get("mode", "inference")).strip().lower()
    if mode not in {"inference", "train"}:
        raise ValueError(f"Unsupported loader mode: {mode}. Expected 'inference' or 'train'.")
    use_group_probs = bool(config_json.get("use_group_probs", True))
    batch_size = max(1, int(config_json.get("batch_size", 64)))
    row_groups_per_chunk = max(1, int(config_json.get("chunk_row_groups", config_json.get("row_groups_per_chunk", 4))))
    num_workers = max(0, int(config_json.get("chunk_workers", config_json.get("num_workers", 0))))
    return mode, use_group_probs, batch_size, row_groups_per_chunk, num_workers


def _count_input_rows(parquet_paths: list[str]) -> int:
    total = 0
    for p in parquet_paths:
        total += int(pq.ParquetFile(p).metadata.num_rows)
    return total


def load_group_splitter_inference_inputs_local(
    parquet_paths: list[str],
    *,
    group_probs_parquet_paths: list[str] | None = None,
    config_json: dict | None = None,
) -> dict:
    cfg = dict(config_json or {})
    mode, use_group_probs, batch_size, row_groups_per_chunk, num_workers = _resolve_inference_runtime(cfg)
    loader = GroupSplitterGraphLoader(
        parquet_paths=[str(Path(p).expanduser().resolve()) for p in parquet_paths],
        group_probs_parquet_paths=[str(Path(p).expanduser().resolve()) for p in group_probs_parquet_paths]
        if group_probs_parquet_paths is not None
        else None,
        mode=mode,
        use_group_probs=use_group_probs,
        batch_size=batch_size,
        row_groups_per_chunk=row_groups_per_chunk,
        num_workers=num_workers,
    )
    dl = loader.make_dataloader(shuffle_batches=False)

    x_parts: list[torch.Tensor] = []
    edge_index_parts: list[torch.Tensor] = []
    edge_attr_parts: list[torch.Tensor] = []
    batch_parts: list[torch.Tensor] = []
    group_total_energy_parts: list[torch.Tensor] = []
    gp_parts: list[torch.Tensor] = []
    y_parts: list[torch.Tensor] = []
    tgroup_parts: list[torch.Tensor] = []
    node_event_parts: list[torch.Tensor] = []

    node_offset = 0
    graph_offset = 0

    for batch in dl:
        x = batch.x.to(torch.float32).cpu()
        edge_index = batch.edge_index.to(torch.int64).cpu()
        edge_attr = batch.edge_attr.to(torch.float32).cpu()
        b = batch.batch.to(torch.int64).cpu()
        group_total_energy = batch.group_total_energy.to(torch.float32).cpu()
        gp = batch.group_probs.to(torch.float32).cpu()
        tgroup = batch.time_group_ids.to(torch.int64).cpu()
        event_ids = batch.event_ids.to(torch.int64).cpu()
        local_counts = torch.bincount(b, minlength=int(batch.num_graphs)).to(torch.int64)
        node_events = event_ids.repeat_interleave(local_counts)

        if x.numel() > 0:
            x_parts.append(x)
            tgroup_parts.append(tgroup)
            node_event_parts.append(node_events)
        if edge_index.numel() > 0:
            edge_index_parts.append(edge_index + int(node_offset))
            edge_attr_parts.append(edge_attr)
        if b.numel() > 0:
            batch_parts.append(b + int(graph_offset))
        group_total_energy_parts.append(group_total_energy)
        gp_parts.append(gp)
        if hasattr(batch, "y") and batch.y is not None:
            y_parts.append(batch.y.to(torch.float32).cpu())

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
    group_total_energy_out = (
        torch.cat(group_total_energy_parts, dim=0) if group_total_energy_parts else torch.empty((0, 1), dtype=torch.float32)
    )
    group_probs_out = torch.cat(gp_parts, dim=0) if gp_parts else torch.empty((0, 3), dtype=torch.float32)
    targets_out = torch.cat(y_parts, dim=0) if y_parts else None
    node_time_groups = torch.cat(tgroup_parts, dim=0) if tgroup_parts else torch.empty((0,), dtype=torch.int64)
    node_event_ids = torch.cat(node_event_parts, dim=0) if node_event_parts else torch.empty((0,), dtype=torch.int64)

    return {
        "x": x_out,
        "edge_index": edge_index_out,
        "edge_attr": edge_attr_out,
        "batch": batch_out,
        "group_total_energy": group_total_energy_out,
        "group_probs": group_probs_out,
        "targets": targets_out,
        "node_time_group_ids": node_time_groups,
        "node_event_ids": node_event_ids,
        "num_rows": _count_input_rows(loader.parquet_paths),
        "validated_files": list(loader.parquet_paths),
        "validated_group_probs_files": list(loader.group_probs_parquet_paths or []),
    }


@step(enable_cache=False)
def load_group_splitter_inference_inputs(
    parquet_paths: list[str],
    group_probs_parquet_paths: list[str] | None = None,
    pipeline_config: dict | None = None,
) -> dict:
    cfg = _resolve_config(pipeline_config)
    config_json = dict(cfg.get("config_json") or {})
    mode, use_group_probs, batch_size, row_groups_per_chunk, num_workers = _resolve_inference_runtime(config_json)
    resolved = [str(Path(p).expanduser().resolve()) for p in parquet_paths]
    if not resolved:
        raise RuntimeError("No parquet paths provided for inference.")
    resolved_probs = [str(Path(p).expanduser().resolve()) for p in group_probs_parquet_paths] if group_probs_parquet_paths else None

    return {
        "input_mode": "group_splitter_graph_loader_inference_v1",
        "mode": mode,
        "use_group_probs": bool(use_group_probs),
        "parquet_paths": resolved,
        "group_probs_parquet_paths": resolved_probs,
        "batch_size": int(batch_size),
        "row_groups_per_chunk": int(row_groups_per_chunk),
        "num_workers": int(num_workers),
        "num_rows": int(_count_input_rows(resolved)),
        "validated_files": list(resolved),
        "validated_group_probs_files": list(resolved_probs or []),
    }
