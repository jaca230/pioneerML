from zenml import step

import torch

from pioneerml.common.loader import GroupSplitterGraphLoader
from pioneerml.pipelines.inference.group_splitting.services import GroupSplitterInferenceInputsService


def load_group_splitter_inference_inputs_local(
    parquet_paths: list[str],
    *,
    group_probs_parquet_paths: list[str] | None = None,
    config_json: dict | None = None,
) -> dict:
    cfg = dict(config_json or {})
    mode, use_group_probs, batch_size, row_groups_per_chunk, num_workers = (
        GroupSplitterInferenceInputsService.resolve_inference_runtime(cfg)
    )
    resolved = GroupSplitterInferenceInputsService.resolve_paths(parquet_paths)
    resolved_probs = GroupSplitterInferenceInputsService.resolve_optional_paths(group_probs_parquet_paths)
    loader = GroupSplitterGraphLoader(
        parquet_paths=resolved,
        group_probs_parquet_paths=resolved_probs,
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
        "num_rows": GroupSplitterInferenceInputsService.count_input_rows(resolved),
        "validated_files": list(resolved),
        "validated_group_probs_files": list(resolved_probs or []),
    }


@step(enable_cache=False)
def load_group_splitter_inference_inputs(
    parquet_paths: list[str],
    group_probs_parquet_paths: list[str] | None = None,
    pipeline_config: dict | None = None,
) -> dict:
    service = GroupSplitterInferenceInputsService(pipeline_config=pipeline_config)
    return service.execute(
        parquet_paths=parquet_paths,
        group_probs_parquet_paths=group_probs_parquet_paths,
    )
