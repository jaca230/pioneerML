from __future__ import annotations

import logging
from collections.abc import Mapping
from typing import Any

import torch

from pioneerml.common.data_loader import LoaderFactory


def apply_compile_runtime_defaults(cfg: dict) -> None:
    matmul_precision = str(cfg["matmul_precision"]).strip().lower()
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


def maybe_compile_model(model, compile_cfg: dict | None, *, context: str = "run"):
    cfg = dict(compile_cfg or {})
    if not bool(cfg["enabled"]):
        return model
    if not hasattr(torch, "compile"):
        print(f"[{context}] torch.compile unavailable; using eager mode.")
        return model

    apply_compile_runtime_defaults(cfg)
    mode = str(cfg["mode"])
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


def merge_nested_dicts(*, base: dict[str, Any], override: Mapping[str, Any] | None) -> dict[str, Any]:
    out = dict(base)
    if override is None:
        return out
    for key, value in dict(override).items():
        if isinstance(value, Mapping) and isinstance(out.get(key), dict):
            out[key] = merge_nested_dicts(base=dict(out[key]), override=value)
        else:
            out[key] = value
    return out


def build_loader_params(
    *,
    cfg: dict[str, Any],
    purpose: str,
    forced_batch_size: int | None = None,
    fallback_purpose: str | None = None,
) -> dict[str, Any]:
    params = LoaderFactory._resolve_loader_params(cfg, purpose=purpose, forced_batch_size=forced_batch_size)
    if fallback_purpose is not None:
        raw_loader_cfg = cfg.get("loader_config")
        if isinstance(raw_loader_cfg, dict):
            if not isinstance(raw_loader_cfg.get(purpose), dict) and isinstance(raw_loader_cfg.get(fallback_purpose), dict):
                params = LoaderFactory._resolve_loader_params(
                    cfg,
                    purpose=fallback_purpose,
                    forced_batch_size=forced_batch_size,
                )
    return params


def build_loader_bundle(
    *,
    loader_factory: LoaderFactory,
    cfg: dict[str, Any],
    purpose: str,
    forced_batch_size: int | None = None,
    fallback_purpose: str | None = None,
    default_shuffle: bool = False,
):
    params = build_loader_params(
        cfg=cfg,
        purpose=purpose,
        forced_batch_size=forced_batch_size,
        fallback_purpose=fallback_purpose,
    )
    provider = loader_factory.build_loader(loader_params=params)
    loader = provider.make_dataloader(shuffle_batches=bool(params.get("shuffle_batches", default_shuffle)))
    return provider, params, loader
