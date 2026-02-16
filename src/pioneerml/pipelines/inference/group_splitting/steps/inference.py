import torch
from zenml import step

from pioneerml.common.loader import GroupSplitterGraphLoader


def _resolve_threshold(pipeline_config: dict | None) -> float:
    if not isinstance(pipeline_config, dict):
        return 0.5
    inf_cfg = pipeline_config.get("inference")
    if isinstance(inf_cfg, dict) and inf_cfg.get("threshold") is not None:
        return float(inf_cfg["threshold"])
    return 0.5


@step(enable_cache=False)
def run_group_splitter_inference(
    model_info: dict,
    inputs: dict,
    pipeline_config: dict | None = None,
) -> dict:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_path = model_info["model_path"]
    scripted = torch.jit.load(model_path, map_location=device)
    scripted.eval()

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
        loader = GroupSplitterGraphLoader(
            parquet_paths=[str(p) for p in inputs["parquet_paths"]],
            group_probs_parquet_paths=[str(p) for p in inputs.get("group_probs_parquet_paths") or []] or None,
            mode=str(inputs.get("mode", "inference")),
            use_group_probs=bool(inputs.get("use_group_probs", True)),
            batch_size=int(inputs["batch_size"]),
            row_groups_per_chunk=int(inputs["row_groups_per_chunk"]),
            num_workers=int(inputs["num_workers"]),
        ).make_dataloader(shuffle_batches=False)

        with torch.no_grad():
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
                probs_parts.append(torch.sigmoid(logits).detach().cpu().to(torch.float32))
                local_counts = torch.bincount(batch.batch.to(torch.int64), minlength=int(batch.num_graphs)).to(torch.int64)
                node_event_id_parts.append(batch.event_ids.to(torch.int64).repeat_interleave(local_counts).cpu())
                node_time_group_parts.append(batch.time_group_ids.to(torch.int64).cpu())
                if hasattr(batch, "y") and batch.y is not None:
                    target_parts.append(batch.y.detach().cpu().to(torch.float32))

    if not probs_parts:
        raise RuntimeError("No inference outputs produced from inputs.")

    probs = torch.cat(probs_parts, dim=0)
    targets = torch.cat(target_parts, dim=0) if target_parts else None
    node_event_ids = torch.cat(node_event_id_parts, dim=0)
    node_time_group_ids = torch.cat(node_time_group_parts, dim=0)

    threshold = _resolve_threshold(pipeline_config)
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
