from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping
from datetime import datetime
import inspect
from pathlib import Path
from typing import Any

import torch


class BaseExporter(ABC):
    def __init__(self, *, config: Mapping[str, Any] | None = None) -> None:
        self.config: dict[str, Any] = dict(config or {})

    @property
    @abstractmethod
    def export_type(self) -> str:
        raise NotImplementedError

    @property
    @abstractmethod
    def artifact_suffix(self) -> str:
        raise NotImplementedError

    def build_paths(
        self,
        *,
        export_dir: str,
        filename_prefix: str,
    ) -> tuple[str, Path, Path]:
        out_dir = Path(export_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        artifact_path = out_dir / f"{filename_prefix}_{timestamp}_{self.artifact_suffix}"
        metadata_path = out_dir / f"{filename_prefix}_{timestamp}_meta.json"
        return timestamp, artifact_path, metadata_path

    def build_example(self, *, loader_provider: Any) -> Any:
        loader = loader_provider.make_dataloader(shuffle_batches=False)
        for batch in loader:
            return batch
        raise RuntimeError("Failed to build export example: export loader yielded no batches.")

    @staticmethod
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

    @staticmethod
    def _to_cpu(value: Any) -> Any:
        if torch.is_tensor(value):
            return value.detach().cpu()
        if isinstance(value, tuple):
            return tuple(BaseExporter._to_cpu(v) for v in value)
        if isinstance(value, list):
            return [BaseExporter._to_cpu(v) for v in value]
        if isinstance(value, dict):
            return {k: BaseExporter._to_cpu(v) for k, v in value.items()}
        return value

    def _resolve_custom_export_fn(self, *, model_obj: Any, function_names: tuple[str, ...]):
        for name in function_names:
            fn = getattr(model_obj, name, None)
            if callable(fn):
                return fn
        return None

    def _call_custom_export(
        self,
        *,
        export_fn,
        output_path: Path,
        example: Any,
        prefer_cuda: bool,
    ) -> None:
        sig = inspect.signature(export_fn)
        kwargs: dict[str, Any] = {}
        if "prefer_cuda" in sig.parameters:
            kwargs["prefer_cuda"] = bool(prefer_cuda)
        for key in ("export_type", "export_mode", "mode"):
            if key in sig.parameters:
                kwargs[key] = self.export_type
                break
        if "example" in sig.parameters and example is not None:
            export_fn(output_path, example, **kwargs)
            return
        export_fn(output_path, **kwargs)

    @abstractmethod
    def export(
        self,
        *,
        model_obj: Any,
        output_path: Path,
        prefer_cuda: bool,
        cfg: Mapping[str, Any],
        dataset: Any,
        loader_provider: Any,
    ) -> None:
        raise NotImplementedError
