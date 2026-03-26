from __future__ import annotations

import logging
from typing import Any

import torch

from .base_compiler import BaseCompiler
from .factory.registry import REGISTRY as COMPILER_REGISTRY


@COMPILER_REGISTRY.register("torch_compile")
class TorchCompileCompiler(BaseCompiler):
    def compile(
        self,
        *,
        model: Any,
        context: str = "run",
    ) -> Any:
        cfg = dict(self.config or {})
        if not bool(cfg.get("enabled", False)):
            return model
        if not hasattr(torch, "compile"):
            print(f"[{context}] torch.compile unavailable; using eager mode.")
            return model

        self._apply_runtime_defaults(cfg)
        mode = str(cfg.get("mode", "default"))
        dynamic = cfg.get("dynamic")
        backend = cfg.get("backend")
        kwargs: dict[str, Any] = {"mode": mode}
        if dynamic is not None:
            kwargs["dynamic"] = bool(dynamic)
        if backend:
            kwargs["backend"] = str(backend)
        try:
            return torch.compile(model, **kwargs)
        except Exception as exc:
            print(f"[{context}] torch.compile failed ({exc}); using eager mode.")
            return model

    @staticmethod
    def _apply_runtime_defaults(cfg: dict[str, Any]) -> None:
        matmul_precision = str(cfg.get("matmul_precision", "")).strip().lower()
        if matmul_precision in {"high", "medium", "highest"}:
            torch.set_float32_matmul_precision(matmul_precision)

        capture_scalars = cfg.get("capture_scalar_outputs")
        if capture_scalars is not None and hasattr(torch, "_dynamo") and hasattr(torch._dynamo, "config"):
            torch._dynamo.config.capture_scalar_outputs = bool(capture_scalars)

        skip_dynamic_cudagraphs = cfg.get("cudagraph_skip_dynamic_graphs")
        if (
            skip_dynamic_cudagraphs is not None
            and hasattr(torch, "_inductor")
            and hasattr(torch._inductor, "config")
            and hasattr(torch._inductor.config, "triton")
        ):
            torch._inductor.config.triton.cudagraph_skip_dynamic_graphs = bool(skip_dynamic_cudagraphs)

        max_autotune = cfg.get("max_autotune")
        if hasattr(torch, "_inductor") and hasattr(torch._inductor, "config"):
            if hasattr(torch._inductor.config, "max_autotune"):
                torch._inductor.config.max_autotune = bool(max_autotune)
            max_autotune_gemm = cfg.get("max_autotune_gemm")
            if hasattr(torch._inductor.config, "max_autotune_gemm"):
                torch._inductor.config.max_autotune_gemm = bool(max_autotune_gemm)

        inductor_log_level = cfg.get("inductor_log_level")
        if inductor_log_level is not None and hasattr(torch, "_logging") and hasattr(torch._logging, "set_logs"):
            level = getattr(logging, str(inductor_log_level).upper(), logging.ERROR)
            torch._logging.set_logs(inductor=level)

