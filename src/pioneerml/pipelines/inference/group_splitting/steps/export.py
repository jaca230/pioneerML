import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch
from zenml import step


def _get_export_cfg(pipeline_config: dict | None) -> dict:
    if isinstance(pipeline_config, dict) and isinstance(pipeline_config.get("export"), dict):
        return dict(pipeline_config["export"])
    return {}


@step(enable_cache=False)
def export_group_splitter_predictions(
    inference_outputs: dict,
    output_dir: str | None = None,
    output_path: str | None = None,
    metrics_path: str | None = None,
    pipeline_config: dict | None = None,
) -> dict:
    cfg = _get_export_cfg(pipeline_config)
    write_timestamped = bool(cfg.get("write_timestamped", False))
    check_accuracy = bool(cfg.get("check_accuracy", False))

    default_dir = Path("data") / "group_splitter"
    out_dir = Path(output_dir) if output_dir else Path(cfg.get("output_dir", default_dir))
    out_dir.mkdir(parents=True, exist_ok=True)

    probs = inference_outputs["probs"]
    targets = inference_outputs.get("targets")
    event_ids = inference_outputs["node_event_ids"]
    time_group_ids = inference_outputs["node_time_group_ids"]
    num_rows = int(inference_outputs["num_rows"])

    probs_np = probs.detach().cpu().numpy().astype(np.float32, copy=False)
    event_ids_np = event_ids.detach().cpu().numpy().astype(np.int64, copy=False)
    tg_np = time_group_ids.detach().cpu().numpy().astype(np.int64, copy=False)

    if num_rows <= 0:
        num_rows = int(event_ids_np.max()) + 1 if event_ids_np.size > 0 else 0

    valid = (event_ids_np >= 0) & (event_ids_np < num_rows) & (tg_np >= 0)
    event_ids_v = event_ids_np[valid]
    tg_v = tg_np[valid]
    probs_v = probs_np[valid]

    if event_ids_v.size > 0:
        order = np.lexsort((tg_v, event_ids_v))
        event_sorted = event_ids_v[order]
        tg_sorted = tg_v[order]
        probs_sorted = probs_v[order]
    else:
        event_sorted = np.zeros((0,), dtype=np.int64)
        tg_sorted = np.zeros((0,), dtype=np.int64)
        probs_sorted = np.zeros((0, 3), dtype=np.float32)

    counts = np.bincount(event_sorted, minlength=num_rows).astype(np.int64, copy=False)
    offsets = np.zeros((num_rows + 1,), dtype=np.int64)
    offsets[1:] = np.cumsum(counts, dtype=np.int64)

    event_id_col = pa.array(np.arange(num_rows, dtype=np.int64))
    time_group_col = pa.ListArray.from_arrays(offsets, pa.array(tg_sorted, type=pa.int64()))
    pred_pion = pa.ListArray.from_arrays(offsets, pa.array(probs_sorted[:, 0], type=pa.float32()))
    pred_muon = pa.ListArray.from_arrays(offsets, pa.array(probs_sorted[:, 1], type=pa.float32()))
    pred_mip = pa.ListArray.from_arrays(offsets, pa.array(probs_sorted[:, 2], type=pa.float32()))
    table = pa.table(
        {
            "event_id": event_id_col,
            "time_group_ids": time_group_col,
            "pred_hit_pion": pred_pion,
            "pred_hit_muon": pred_muon,
            "pred_hit_mip": pred_mip,
        }
    )

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    validated_files = [str(Path(p).expanduser().resolve()) for p in inference_outputs.get("validated_files", [])]
    validated_group_probs_files = [
        str(Path(p).expanduser().resolve()) for p in inference_outputs.get("validated_group_probs_files", [])
    ]
    if output_path:
        latest_pred_path = Path(output_path)
    else:
        if len(validated_files) == 1:
            stem = Path(validated_files[0]).stem
            latest_pred_path = out_dir / f"{stem}_preds_latest.parquet"
        else:
            latest_pred_path = out_dir / "group_splitter_preds_latest.parquet"
    latest_pred_path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, latest_pred_path)

    timestamped_pred_path = None
    if write_timestamped:
        if len(validated_files) == 1:
            stem = Path(validated_files[0]).stem
            p = out_dir / f"{stem}_preds_{ts}.parquet"
        else:
            p = out_dir / f"group_splitter_preds_{ts}.parquet"
        pq.write_table(table, p)
        timestamped_pred_path = str(p)

    metrics = {
        "mode": "group_splitter",
        "model_path": inference_outputs["model_path"],
        "output_path": str(latest_pred_path),
        "threshold": float(inference_outputs["threshold"]),
        "validated_files": validated_files,
        "validated_group_probs_files": validated_group_probs_files,
        "loss": None,
    }

    if check_accuracy and targets is not None and targets.numel() > 0:
        preds_binary = inference_outputs["preds_binary"].to(torch.float32)
        t = targets.to(torch.float32)
        metrics["accuracy"] = float((preds_binary == t).to(torch.float32).mean().item())
        metrics["exact_match"] = float(((preds_binary == t).all(dim=1)).to(torch.float32).mean().item())
        confusion = []
        for cls in range(int(t.shape[1])):
            truth = t[:, cls].to(torch.int64)
            pred = preds_binary[:, cls].to(torch.int64)
            fp = int(((truth == 0) & (pred == 1)).sum().item())
            fn = int(((truth == 1) & (pred == 0)).sum().item())
            tp = int(((truth == 1) & (pred == 1)).sum().item())
            total = float(tp + fp + fn)
            if total > 0:
                confusion.append({"tp": tp / total, "fp": fp / total, "fn": fn / total})
            else:
                confusion.append({"tp": 0.0, "fp": 0.0, "fn": 0.0})
        metrics["confusion"] = confusion
    elif check_accuracy:
        metrics["accuracy"] = None
        metrics["exact_match"] = None
        metrics["confusion"] = None

    latest_metrics = Path(metrics_path) if metrics_path else out_dir / "metrics_latest.json"
    latest_metrics.parent.mkdir(parents=True, exist_ok=True)
    latest_metrics.write_text(json.dumps(metrics, indent=2, sort_keys=True), encoding="utf-8")

    timestamped_metrics_path = None
    if write_timestamped:
        p = out_dir / f"metrics_{ts}.json"
        p.write_text(json.dumps(metrics, indent=2, sort_keys=True), encoding="utf-8")
        timestamped_metrics_path = str(p)

    return {
        "predictions_path": str(latest_pred_path),
        "metrics_path": str(latest_metrics),
        "timestamped_predictions_path": timestamped_pred_path,
        "timestamped_metrics_path": timestamped_metrics_path,
        "num_rows": num_rows,
    }
