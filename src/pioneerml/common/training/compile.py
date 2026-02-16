from __future__ import annotations

import logging

import torch


def _apply_compile_runtime_defaults(cfg: dict) -> None:
    # Dynamic graph workloads (PyG) often benefit from these defaults.
    matmul_precision = str(cfg.get("matmul_precision", "high")).strip().lower()
    if matmul_precision in {"high", "medium", "highest"}:
        torch.set_float32_matmul_precision(matmul_precision)

    capture_scalars = cfg.get("capture_scalar_outputs", True)
    if capture_scalars is not None and hasattr(torch, "_dynamo") and hasattr(torch._dynamo, "config"):
        torch._dynamo.config.capture_scalar_outputs = bool(capture_scalars)

    skip_dynamic_cudagraphs = cfg.get("cudagraph_skip_dynamic_graphs", True)
    if (
        skip_dynamic_cudagraphs is not None
        and hasattr(torch, "_inductor")
        and hasattr(torch._inductor, "config")
        and hasattr(torch._inductor.config, "triton")
    ):
        torch._inductor.config.triton.cudagraph_skip_dynamic_graphs = bool(skip_dynamic_cudagraphs)

    # Disable expensive autotune paths by default for dynamic graph workloads.
    max_autotune = cfg.get("max_autotune", False)
    if hasattr(torch, "_inductor") and hasattr(torch._inductor, "config"):
        if hasattr(torch._inductor.config, "max_autotune"):
            torch._inductor.config.max_autotune = bool(max_autotune)
        max_autotune_gemm = cfg.get("max_autotune_gemm", bool(max_autotune))
        if hasattr(torch._inductor.config, "max_autotune_gemm"):
            torch._inductor.config.max_autotune_gemm = bool(max_autotune_gemm)

    # Reduce noisy inductor warnings by default (override with inductor_log_level).
    inductor_log_level = cfg.get("inductor_log_level", "ERROR")
    if inductor_log_level is not None and hasattr(torch, "_logging") and hasattr(torch._logging, "set_logs"):
        level = getattr(logging, str(inductor_log_level).upper(), logging.ERROR)
        torch._logging.set_logs(inductor=level)


def maybe_compile_model(model, compile_cfg: dict | None, *, context: str = "train"):
    cfg = dict(compile_cfg or {})
    if not bool(cfg.get("enabled", False)):
        return model
    if not hasattr(torch, "compile"):
        print(f"[{context}] torch.compile unavailable; using eager mode.")
        return model
    _apply_compile_runtime_defaults(cfg)

    mode = str(cfg.get("mode", "default"))
    dynamic = cfg.get("dynamic")
    backend = cfg.get("backend")
    kwargs = {"mode": mode}
    if dynamic is not None:
        kwargs["dynamic"] = bool(dynamic)
    if backend:
        kwargs["backend"] = str(backend)

    try:
        return torch.compile(model, **kwargs)
    except Exception as exc:
        print(f"[{context}] torch.compile failed ({exc}); using eager mode.")
        return model


def restore_eager_model_if_compiled(module) -> None:
    model_obj = getattr(module, "model", None)
    if model_obj is not None and hasattr(model_obj, "_orig_mod"):
        module.model = model_obj._orig_mod
