from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any
import torch
import inspect


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


def _resolve_custom_export_fn(*, model_obj: Any, export_type: str):
    by_type = {
        "script": ("export_torchscript", "export_script"),
        "trace": ("export_torchscript", "export_trace"),
        "torch_export": ("export_torch_export", "export_exported_program"),
    }
    for name in by_type.get(export_type, ()):
        fn = getattr(model_obj, name, None)
        if callable(fn):
            return fn
    return None


def build_example_from_loader_provider(*, loader_provider: Any) -> Any:
    loader = loader_provider.make_dataloader(shuffle_batches=False)
    for batch in loader:
        return batch
    raise RuntimeError("Failed to build export example: export loader yielded no batches.")


def _normalize_example_for_inputs(*, example: Any) -> tuple[tuple[Any, ...], dict[str, Any]]:
    if isinstance(example, dict) and ("args" in example or "kwargs" in example):
        args = example.get("args", ())
        kwargs = example.get("kwargs", {})
        if not isinstance(args, tuple):
            if isinstance(args, list):
                args = tuple(args)
            else:
                args = (args,)
        if not isinstance(kwargs, dict):
            raise TypeError("Export example kwargs must be a dict.")
        return args, kwargs
    if isinstance(example, tuple):
        return example, {}
    if isinstance(example, list):
        return tuple(example), {}
    return (example,), {}


def _to_cpu(value: Any) -> Any:
    if torch.is_tensor(value):
        return value.detach().cpu()
    if isinstance(value, tuple):
        return tuple(_to_cpu(v) for v in value)
    if isinstance(value, list):
        return [_to_cpu(v) for v in value]
    if isinstance(value, dict):
        return {k: _to_cpu(v) for k, v in value.items()}
    return value


def _call_custom_export(
    *,
    export_fn,
    output_path: Path,
    export_type: str,
    prefer_cuda: bool,
    example: Any,
) -> None:
    sig = inspect.signature(export_fn)
    kwargs: dict[str, Any] = {}
    if "prefer_cuda" in sig.parameters:
        kwargs["prefer_cuda"] = bool(prefer_cuda)
    for key in ("export_type", "export_mode", "mode"):
        if key in sig.parameters:
            kwargs[key] = export_type
            break

    if "example" in sig.parameters and example is not None:
        export_fn(output_path, example, **kwargs)
        return

    export_fn(output_path, **kwargs)


def export_model_artifact(
    *,
    model_obj: Any,
    output_path: Path,
    export_type: str,
    prefer_cuda: bool,
    cfg: dict,
    dataset,
    loader_provider: Any,
) -> None:
    if model_obj is None:
        raise RuntimeError("Cannot export: model object is missing.")

    _ = cfg
    _ = dataset
    example = build_example_from_loader_provider(loader_provider=loader_provider)

    custom_fn = _resolve_custom_export_fn(model_obj=model_obj, export_type=export_type)
    if callable(custom_fn):
        _call_custom_export(
            export_fn=custom_fn,
            output_path=output_path,
            export_type=export_type,
            prefer_cuda=prefer_cuda,
            example=example,
        )
        return

    model = model_obj.eval()
    if hasattr(model, "to"):
        model = model.to("cpu")

    if export_type == "script":
        scripted = torch.jit.script(model)
        scripted.save(str(output_path))
        return

    if export_type == "trace":
        args, _ = _normalize_example_for_inputs(example=_to_cpu(example))
        traced = torch.jit.trace(model, args)
        traced.save(str(output_path))
        return

    if export_type == "torch_export":
        if not hasattr(torch, "export") or not hasattr(torch.export, "export") or not hasattr(torch.export, "save"):
            raise RuntimeError("torch.export is unavailable in this runtime.")
        args, kwargs = _normalize_example_for_inputs(example=_to_cpu(example))
        program = torch.export.export(model, args=args, kwargs=kwargs)
        torch.export.save(program, str(output_path))
        return

    allowed = "script, trace, torch_export"
    raise ValueError(f"Unsupported export type: {export_type!r}. Expected one of [{allowed}].")


def build_export_paths(*, export_dir: str, filename_prefix: str, export_type: str) -> tuple[str, Path, Path]:
    out_dir = Path(export_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if export_type == "script":
        suffix = "torchscript.pt"
    elif export_type == "trace":
        suffix = "trace.pt"
    elif export_type == "torch_export":
        suffix = "export.pt2"
    else:
        suffix = "artifact.bin"
    torchscript_path = out_dir / f"{filename_prefix}_{timestamp}_{suffix}"
    metadata_path = out_dir / f"{filename_prefix}_{timestamp}_meta.json"
    return timestamp, torchscript_path, metadata_path


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
