from zenml import step

import torch

from pioneerml.common.loader import GroupClassifierGraphLoader
from pioneerml.pipelines.inference.group_classification.services import GroupClassifierInferenceInputsService


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
    mode, batch_size, row_groups_per_chunk, num_workers = GroupClassifierInferenceInputsService.resolve_inference_runtime(cfg)
    resolved = GroupClassifierInferenceInputsService.resolve_paths(parquet_paths)
    loader = GroupClassifierGraphLoader(
        parquet_paths=resolved,
        mode=mode,
        batch_size=batch_size,
        row_groups_per_chunk=row_groups_per_chunk,
        num_workers=num_workers,
    ).make_dataloader(shuffle_batches=False)

    x_parts: list[torch.Tensor] = []
    edge_index_parts: list[torch.Tensor] = []
    edge_attr_parts: list[torch.Tensor] = []
    batch_parts: list[torch.Tensor] = []
    graph_event_id_parts: list[torch.Tensor] = []
    graph_group_id_parts: list[torch.Tensor] = []

    node_offset = 0
    graph_offset = 0
    for batch in loader:
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
    edge_index_out = torch.cat(edge_index_parts, dim=1) if edge_index_parts else torch.empty((2, 0), dtype=torch.int64)
    edge_attr_out = torch.cat(edge_attr_parts, dim=0) if edge_attr_parts else torch.empty((0, 4), dtype=torch.float32)
    batch_out = torch.cat(batch_parts, dim=0) if batch_parts else torch.empty((0,), dtype=torch.int64)
    graph_event_ids = (
        torch.cat(graph_event_id_parts, dim=0) if graph_event_id_parts else torch.empty((0,), dtype=torch.int64)
    )
    graph_group_ids = (
        torch.cat(graph_group_id_parts, dim=0) if graph_group_id_parts else torch.empty((0,), dtype=torch.int64)
    )
    targets_out = torch.zeros((graph_offset, 3), dtype=torch.float32)
    return {
        "x": x_out,
        "edge_index": edge_index_out,
        "edge_attr": edge_attr_out,
        "batch": batch_out,
        "targets": targets_out,
        "num_rows": GroupClassifierInferenceInputsService.count_input_rows(resolved),
        "graph_event_ids": graph_event_ids,
        "graph_group_ids": graph_group_ids,
        "validated_files": resolved,
    }


@step(enable_cache=False)
def load_group_classifier_inference_inputs(
    parquet_paths: list[str],
    pipeline_config: dict | None = None,
) -> dict:
    service = GroupClassifierInferenceInputsService(pipeline_config=pipeline_config)
    return service.execute(parquet_paths=parquet_paths)
