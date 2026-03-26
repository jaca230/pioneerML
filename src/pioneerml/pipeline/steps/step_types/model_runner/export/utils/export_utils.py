from __future__ import annotations

import json
from typing import Any
from pathlib import Path


def json_safe(value):
    if value is None:
        return None
    try:
        json.dumps(value)
        return value
    except TypeError:
        if isinstance(value, dict):
            return {str(key): json_safe(item) for key, item in value.items()}
        if isinstance(value, (list, tuple)):
            return [json_safe(item) for item in value]
        return str(value)


def resolve_tensor_last_dim(obj, *names: str) -> int:
    for name in names:
        tensor = getattr(obj, name, None)
        if tensor is not None and hasattr(tensor, "shape") and len(tensor.shape) >= 2:
            return int(tensor.shape[-1])
    return 0


def build_data_shapes(*, dataset: Any) -> dict[str, int]:
    bundle_inputs = getattr(dataset, "inputs", None)
    if bundle_inputs is None:
        bundle_inputs = getattr(dataset, "data", None)

    bundle_targets = getattr(dataset, "targets", None)
    if bundle_targets is None and bundle_inputs is not None:
        bundle_targets = getattr(bundle_inputs, "y_graph", None)
        if bundle_targets is None:
            bundle_targets = getattr(bundle_inputs, "y_node", None)
        if bundle_targets is None:
            bundle_targets = getattr(bundle_inputs, "y", None)

    x_dim = resolve_tensor_last_dim(bundle_inputs, "x_node", "x")
    edge_attr_dim = resolve_tensor_last_dim(bundle_inputs, "x_edge", "edge_attr")
    num_classes = 0
    if bundle_targets is not None and hasattr(bundle_targets, "shape") and len(bundle_targets.shape) >= 2:
        num_classes = int(bundle_targets.shape[-1])
    return {
        "x_dim": int(x_dim),
        "edge_attr_dim": int(edge_attr_dim),
        "num_classes": int(num_classes),
    }


def write_export_metadata(*, metadata_path: Path, metadata: dict[str, Any]) -> None:
    with metadata_path.open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2, sort_keys=True)
