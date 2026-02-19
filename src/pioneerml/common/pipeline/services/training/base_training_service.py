from __future__ import annotations

import logging
import shutil
import tempfile
from typing import Callable

import pytorch_lightning as pl
import torch
import torch.nn as nn

from ..base_pipeline_service import BasePipelineService
from pioneerml.common.zenml.utils import detect_available_accelerator
from .utils import GraphLightningModule, LightningWarningFilter, RelativeEarlyStopping


class BaseTrainingService(BasePipelineService):
    RelativeEarlyStopping = RelativeEarlyStopping

    def __init__(self, *, pipeline_config: dict | None = None) -> None:
        super().__init__(pipeline_config=pipeline_config)
        self._warning_filter = LightningWarningFilter()

    def apply_warning_filter(self) -> None:
        self._warning_filter.apply_default()

    @staticmethod
    def set_tensor_core_precision(mode: str = "medium") -> str | None:
        if not torch.cuda.is_available():
            return None
        if mode not in {"high", "medium"}:
            mode = "medium"
        torch.set_float32_matmul_precision(mode)
        return mode

    @staticmethod
    def default_precision_for_accelerator(accelerator: str | None) -> str:
        if accelerator in {"cuda", "gpu", "auto"} and torch.cuda.is_available():
            return "16-mixed"
        return "32-true"

    def apply_precision(self, *, accelerator: str | None = None, mode: str | None = None) -> str:
        resolved = mode or self.default_precision_for_accelerator(accelerator)
        applied = self.set_tensor_core_precision(resolved)
        return applied or str(resolved)

    @staticmethod
    def _apply_compile_runtime_defaults(cfg: dict) -> None:
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

        max_autotune = cfg.get("max_autotune", False)
        if hasattr(torch, "_inductor") and hasattr(torch._inductor, "config"):
            if hasattr(torch._inductor.config, "max_autotune"):
                torch._inductor.config.max_autotune = bool(max_autotune)
            max_autotune_gemm = cfg.get("max_autotune_gemm", bool(max_autotune))
            if hasattr(torch._inductor.config, "max_autotune_gemm"):
                torch._inductor.config.max_autotune_gemm = bool(max_autotune_gemm)

        inductor_log_level = cfg.get("inductor_log_level", "ERROR")
        if inductor_log_level is not None and hasattr(torch, "_logging") and hasattr(torch._logging, "set_logs"):
            level = getattr(logging, str(inductor_log_level).upper(), logging.ERROR)
            torch._logging.set_logs(inductor=level)

    @staticmethod
    def maybe_compile_model(model, compile_cfg: dict | None, *, context: str = "train"):
        cfg = dict(compile_cfg or {})
        if not bool(cfg.get("enabled", False)):
            return model
        if not hasattr(torch, "compile"):
            print(f"[{context}] torch.compile unavailable; using eager mode.")
            return model
        BaseTrainingService._apply_compile_runtime_defaults(cfg)

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

    @staticmethod
    def restore_eager_model_if_compiled(module) -> None:
        model_obj = getattr(module, "model", None)
        if model_obj is not None and hasattr(model_obj, "_orig_mod"):
            module.model = model_obj._orig_mod

    def compile_model(self, model, *, compile_cfg: dict | None, context: str) -> nn.Module:
        return self.maybe_compile_model(model, compile_cfg, context=context)

    @staticmethod
    def build_early_stopping_callback(cfg: dict | None) -> pl.callbacks.EarlyStopping | None:
        config = dict(cfg or {})
        if not bool(config.get("enabled", False)):
            return None

        monitor = str(config.get("monitor", "val_loss"))
        mode = str(config.get("mode", "min"))
        patience = int(config.get("patience", 5))
        min_delta = float(config.get("min_delta", 0.0))
        strict = bool(config.get("strict", True))
        check_finite = bool(config.get("check_finite", True))
        verbose = bool(config.get("verbose", False))
        delta_mode = str(config.get("min_delta_mode", "absolute")).strip().lower()

        common_kwargs = {
            "monitor": monitor,
            "mode": mode,
            "patience": patience,
            "strict": strict,
            "check_finite": check_finite,
            "verbose": verbose,
        }
        if delta_mode in {"relative", "percent", "pct"}:
            return RelativeEarlyStopping(relative_min_delta=min_delta, min_delta=0.0, **common_kwargs)
        return pl.callbacks.EarlyStopping(min_delta=min_delta, **common_kwargs)

    def build_early_stopping(self, cfg: dict | None):
        return self.build_early_stopping_callback(cfg)

    def build_graph_module(
        self,
        *,
        model: nn.Module,
        task: str,
        loss_fn: Callable | nn.Module | None,
        lr: float,
        weight_decay: float,
        threshold: float = 0.5,
        scheduler_step_size: int | None = None,
        scheduler_gamma: float = 0.5,
    ) -> GraphLightningModule:
        return GraphLightningModule(
            model,
            task=task,
            loss_fn=loss_fn,
            lr=float(lr),
            weight_decay=float(weight_decay),
            threshold=float(threshold),
            scheduler_step_size=(int(scheduler_step_size) if scheduler_step_size is not None else None),
            scheduler_gamma=float(scheduler_gamma),
        )

    def fit_module(
        self,
        *,
        module,
        train_loader,
        val_loader,
        max_epochs: int,
        grad_clip: float | None,
        trainer_kwargs: dict | None,
        early_stopping_cfg: dict | None,
    ):
        return self._fit_with_loaders(
            module=module,
            train_loader=train_loader,
            val_loader=val_loader,
            max_epochs=max_epochs,
            grad_clip=grad_clip,
            trainer_kwargs=trainer_kwargs,
            early_stopping_cfg=early_stopping_cfg,
        )

    def _fit_with_loaders(
        self,
        *,
        module,
        train_loader,
        val_loader,
        max_epochs: int,
        grad_clip: float | None,
        trainer_kwargs: dict | None,
        early_stopping_cfg: dict | None,
    ):
        merged_trainer_kwargs = dict(trainer_kwargs or {})
        if grad_clip is not None:
            merged_trainer_kwargs.setdefault("gradient_clip_val", float(grad_clip))

        es_cfg = dict(early_stopping_cfg or {})
        monitor = str(es_cfg.get("monitor", "val_loss"))
        mode = str(es_cfg.get("mode", "min"))

        callbacks = list(merged_trainer_kwargs.get("callbacks") or [])
        ckpt_dir = tempfile.mkdtemp(prefix="pioneerml_best_ckpt_")
        ckpt_callback = pl.callbacks.ModelCheckpoint(
            dirpath=ckpt_dir,
            filename="best",
            monitor=monitor,
            mode=mode,
            save_top_k=1,
            save_last=False,
            auto_insert_metric_name=False,
        )
        callbacks.append(ckpt_callback)
        es_callback = self.build_early_stopping_callback(early_stopping_cfg)
        if es_callback is not None:
            callbacks.append(es_callback)
        if callbacks:
            merged_trainer_kwargs["callbacks"] = callbacks

        accelerator, devices = detect_available_accelerator()
        trainer = None
        try:
            trainer = pl.Trainer(
                accelerator=accelerator,
                devices=devices,
                max_epochs=int(max_epochs),
                enable_checkpointing=True,
                logger=False,
                **merged_trainer_kwargs,
            )
            trainer.fit(module, train_dataloaders=train_loader, val_dataloaders=val_loader)

            best_model_path = str(getattr(ckpt_callback, "best_model_path", "") or "")
            if best_model_path:
                best_ckpt = torch.load(best_model_path, map_location="cpu")
                state_dict = best_ckpt.get("state_dict", best_ckpt)
                module.load_state_dict(state_dict, strict=True)
        finally:
            self.restore_eager_model_if_compiled(module)
            shutil.rmtree(ckpt_dir, ignore_errors=True)
        return module
