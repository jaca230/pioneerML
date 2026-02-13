import torch
from zenml import step

from pioneerml.common.loader import GroupClassifierGraphLoader


def _resolve_threshold(pipeline_config: dict | None) -> float:
    if not isinstance(pipeline_config, dict):
        return 0.5
    inf_cfg = pipeline_config.get("inference")
    if isinstance(inf_cfg, dict) and inf_cfg.get("threshold") is not None:
        return float(inf_cfg["threshold"])
    return 0.5


def _scripted_expects_u(scripted: torch.jit.ScriptModule) -> bool:
    try:
        schema = str(scripted.forward.schema)
    except Exception:
        return False
    return " Tensor u" in schema or ", Tensor u" in schema


def _run_scripted_forward(
    scripted: torch.jit.ScriptModule,
    *,
    expects_u: bool,
    x: torch.Tensor,
    edge_index: torch.Tensor,
    edge_attr: torch.Tensor,
    batch: torch.Tensor,
    num_graphs: int,
    u: torch.Tensor | None = None,
) -> torch.Tensor:
    if expects_u:
        if u is None:
            u = torch.zeros((num_graphs, 1), dtype=torch.float32, device=x.device)
        else:
            u = u.to(dtype=torch.float32, device=x.device)
        return scripted(x, edge_index, edge_attr, batch, u)
    return scripted(x, edge_index, edge_attr, batch)


@step(enable_cache=False)
def run_group_classifier_inference(
    model_info: dict,
    inputs: dict,
    pipeline_config: dict | None = None,
) -> dict:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_path = model_info["model_path"]
    scripted = torch.jit.load(model_path, map_location=device)
    scripted.eval()
    expects_u = _scripted_expects_u(scripted)

    probs_parts: list[torch.Tensor] = []
    target_parts: list[torch.Tensor] = []
    graph_event_id_parts: list[torch.Tensor] = []
    graph_group_id_parts: list[torch.Tensor] = []

    if "x" in inputs:
        x = inputs["x"].to(device)
        edge_index = inputs["edge_index"].to(device)
        edge_attr = inputs["edge_attr"].to(device)
        batch = inputs["batch"].to(device)
        num_graphs = int(inputs["targets"].shape[0]) if inputs.get("targets") is not None else int(batch.max().item() + 1)
        u = inputs.get("u")
        if u is not None:
            u = u.to(device)

        with torch.no_grad():
            logits = _run_scripted_forward(
                scripted,
                expects_u=expects_u,
                x=x,
                edge_index=edge_index,
                edge_attr=edge_attr,
                batch=batch,
                num_graphs=num_graphs,
                u=u,
            )
            if isinstance(logits, (tuple, list)):
                logits = logits[0]
            probs_parts.append(torch.sigmoid(logits).detach().cpu().to(torch.float32))

        if inputs.get("targets") is not None:
            target_parts.append(inputs["targets"].to(torch.float32))
        graph_event_id_parts.append(inputs["graph_event_ids"].to(torch.int64))
        graph_group_id_parts.append(inputs["graph_group_ids"].to(torch.int64))
    else:
        loader = GroupClassifierGraphLoader(
            parquet_paths=[str(p) for p in inputs["parquet_paths"]],
            mode=str(inputs.get("mode", "inference")),
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
                logits = _run_scripted_forward(
                    scripted,
                    expects_u=expects_u,
                    x=x,
                    edge_index=edge_index,
                    edge_attr=edge_attr,
                    batch=b,
                    num_graphs=int(batch.num_graphs),
                )
                if isinstance(logits, (tuple, list)):
                    logits = logits[0]
                probs_parts.append(torch.sigmoid(logits).detach().cpu().to(torch.float32))
                graph_event_id_parts.append(batch.event_ids.to(torch.int64).cpu())
                graph_group_id_parts.append(batch.group_ids.to(torch.int64).cpu())
                if hasattr(batch, "y") and batch.y is not None:
                    target_parts.append(batch.y.detach().cpu().to(torch.float32))

    if not probs_parts:
        raise RuntimeError("No inference outputs produced from inputs.")

    probs = torch.cat(probs_parts, dim=0)
    targets = torch.cat(target_parts, dim=0) if target_parts else None
    graph_event_ids = torch.cat(graph_event_id_parts, dim=0)
    graph_group_ids = torch.cat(graph_group_id_parts, dim=0)

    threshold = _resolve_threshold(pipeline_config)
    preds_binary = (probs >= threshold).to(torch.float32)

    return {
        "probs": probs,
        "preds_binary": preds_binary,
        "targets": targets,
        "graph_event_ids": graph_event_ids,
        "graph_group_ids": graph_group_ids,
        "num_rows": int(inputs.get("num_rows", 0)),
        "validated_files": list(inputs.get("validated_files") or []),
        "threshold": float(threshold),
        "model_path": model_path,
    }
