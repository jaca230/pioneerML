import json
from datetime import datetime
from pathlib import Path
import inspect

from zenml import step

from pioneerml.pipelines.training.group_classification.dataset import GroupClassifierDataset
from pioneerml.pipelines.training.group_classification.steps.config import resolve_step_config


@step
def export_group_classifier(
    module,
    dataset: GroupClassifierDataset,
    pipeline_config: dict | None = None,
    hpo_params: dict | None = None,
    metrics: dict | None = None,
) -> dict:
    step_config = resolve_step_config(pipeline_config, "export")
    cfg = step_config or {}
    if cfg.get("enabled") is False:
        return {"torchscript_path": None, "metadata_path": None, "skipped": True}
    export_dir = Path(cfg.get("export_dir", "trained_models/groupclassifier"))
    export_dir.mkdir(parents=True, exist_ok=True)
    prefix = cfg.get("filename_prefix", "groupclassifier")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    torchscript_path = export_dir / f"{prefix}_{timestamp}_torchscript.pt"
    meta_path = export_dir / f"{prefix}_{timestamp}_meta.json"

    prefer_cuda = bool(cfg.get("prefer_cuda", True))
    export_fn = getattr(module.model, "export_torchscript", None)
    if export_fn is None:
        return {"torchscript_path": None, "metadata_path": None, "skipped": True}
    try:
        sig = inspect.signature(export_fn)
        example = cfg.get("example")
        if example is None and "example" in sig.parameters:
            data = dataset.data
            if hasattr(data, "batch") and hasattr(data, "u"):
                example = (data.x, data.edge_index, data.edge_attr, data.batch, data.u)
        if "example" in sig.parameters:
            if "prefer_cuda" in sig.parameters:
                export_fn(torchscript_path, example, prefer_cuda=prefer_cuda)
            else:
                export_fn(torchscript_path, example)
        else:
            if "prefer_cuda" in sig.parameters:
                export_fn(torchscript_path, prefer_cuda=prefer_cuda)
            else:
                export_fn(torchscript_path)
    except TypeError:
        try:
            export_fn(torchscript_path)
        except TypeError:
            return {"torchscript_path": None, "metadata_path": None, "skipped": True}
    except NotImplementedError:
        return {"torchscript_path": None, "metadata_path": None, "skipped": True}

    meta = {
        "timestamp": timestamp,
        "torchscript_path": str(torchscript_path),
        "hpo_params": hpo_params or {},
        "metrics": metrics or {},
        "pipeline_config": cfg.get("pipeline_config"),
        "data_shapes": {
            "x_dim": int(dataset.data.x.shape[-1]),
            "edge_attr_dim": int(dataset.data.edge_attr.shape[-1]),
            "num_classes": int(dataset.targets.shape[-1]),
        },
    }
    with meta_path.open("w", encoding="utf-8") as handle:
        json.dump(meta, handle, indent=2, sort_keys=True)

    return {
        "torchscript_path": str(torchscript_path),
        "metadata_path": str(meta_path),
    }
