from __future__ import annotations

import inspect
import json
from typing import Callable


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


def call_export(
    *,
    export_fn,
    torchscript_path,
    prefer_cuda: bool,
    cfg: dict,
    dataset,
    example_builder: Callable | None,
) -> None:
    sig = inspect.signature(export_fn)
    example = cfg.get("example")
    if example is None and example_builder is not None and "example" in sig.parameters:
        example = example_builder(dataset)

    if "example" in sig.parameters:
        if "prefer_cuda" in sig.parameters:
            export_fn(torchscript_path, example, prefer_cuda=prefer_cuda)
        else:
            export_fn(torchscript_path, example)
        return
    if "prefer_cuda" in sig.parameters:
        export_fn(torchscript_path, prefer_cuda=prefer_cuda)
        return
    export_fn(torchscript_path)
