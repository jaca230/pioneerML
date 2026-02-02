import json
from datetime import datetime
from pathlib import Path

from zenml import step

from pioneerml.zenml.pipelines.training.group_classification.batch import GroupClassifierBatch


@step
def export_group_classifier(
    module,
    batch: GroupClassifierBatch,
    *,
    step_config: dict | None = None,
    hpo_params: dict | None = None,
    metrics: dict | None = None,
) -> dict:
    cfg = step_config or {}
    export_dir = Path(cfg.get("export_dir", "trained_models/groupclassifier"))
    export_dir.mkdir(parents=True, exist_ok=True)
    prefix = cfg.get("filename_prefix", "groupclassifier")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    torchscript_path = export_dir / f"{prefix}_{timestamp}_torchscript.pt"
    meta_path = export_dir / f"{prefix}_{timestamp}_meta.json"

    prefer_cuda = bool(cfg.get("prefer_cuda", True))
    module.model.export_torchscript(torchscript_path, prefer_cuda=prefer_cuda)

    meta = {
        "timestamp": timestamp,
        "torchscript_path": str(torchscript_path),
        "hpo_params": hpo_params or {},
        "metrics": metrics or {},
        "pipeline_config": cfg.get("pipeline_config"),
        "data_shapes": {
            "x_dim": int(batch.data.x.shape[-1]),
            "edge_attr_dim": int(batch.data.edge_attr.shape[-1]),
            "num_classes": int(batch.targets.shape[-1]),
        },
    }
    with meta_path.open("w", encoding="utf-8") as handle:
        json.dump(meta, handle, indent=2, sort_keys=True)

    return {
        "torchscript_path": str(torchscript_path),
        "metadata_path": str(meta_path),
    }
