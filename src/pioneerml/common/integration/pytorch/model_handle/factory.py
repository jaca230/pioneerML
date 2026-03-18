from __future__ import annotations

from .base_model_handle import BaseModelHandle
# Import modules for decorator side-effects (registration at import-time).
from . import torchexport_model_handle as _torchexport_model_handle
from . import torchscript_model_handle as _torchscript_model_handle
from . import torchtrace_model_handle as _torchtrace_model_handle


class ModelHandleFactory:
    @staticmethod
    def build(*, model_type: str, model_path: str) -> BaseModelHandle:
        _ = (_torchscript_model_handle, _torchtrace_model_handle, _torchexport_model_handle)
        key = str(model_type).strip().lower()
        handle_cls = BaseModelHandle.registry().get(key)
        if handle_cls is None:
            allowed = ", ".join(sorted(BaseModelHandle.registry().keys()))
            raise ValueError(f"Unsupported model handle type '{model_type}'. Allowed: {allowed}")
        return handle_cls(model_path=str(model_path))
