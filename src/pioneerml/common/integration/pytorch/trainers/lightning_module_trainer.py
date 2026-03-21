from __future__ import annotations

import inspect
import shutil
import tempfile
from typing import Any

import pytorch_lightning as pl
import torch

from pioneerml.common.integration.zenml.utils import detect_available_accelerator

from .callbacks.early_stopping import build_early_stopping_callback
from .factory.registry import REGISTRY as TRAINER_REGISTRY


@TRAINER_REGISTRY.register("lightning_module")
class LightningModuleTrainer(pl.Trainer):
    def __init__(
        self,
        *,
        trainer_kwargs: dict[str, Any] | None = None,
        early_stopping_cfg: dict[str, Any] | None = None,
    ) -> None:
        self.early_stopping_cfg = dict(early_stopping_cfg or {})
        self._restore_eager_after_fit = True
        self._checkpoint_dir = tempfile.mkdtemp(prefix="pioneerml_best_ckpt_")
        self._checkpoint_callback = self._build_checkpoint_callback(early_stopping_cfg=self.early_stopping_cfg)
        self._early_stopping_callback = build_early_stopping_callback(early_stopping_cfg=self.early_stopping_cfg)

        kwargs = dict(trainer_kwargs or {})
        callbacks = list(kwargs.get("callbacks") or [])
        callbacks.append(self._checkpoint_callback)
        if self._early_stopping_callback is not None:
            callbacks.append(self._early_stopping_callback)
        if callbacks:
            kwargs["callbacks"] = callbacks

        detected_accelerator, detected_devices = detect_available_accelerator()
        accelerator = kwargs.pop("accelerator", detected_accelerator)
        devices = kwargs.pop("devices", detected_devices)
        max_epochs = int(kwargs.pop("max_epochs", 1))

        super().__init__(
            accelerator=accelerator,
            devices=devices,
            max_epochs=max_epochs,
            enable_checkpointing=True,
            logger=False,
            **kwargs,
        )

    @staticmethod
    def _restore_eager_model_if_compiled(module) -> None:
        model_obj = getattr(module, "model", None)
        if model_obj is not None and hasattr(model_obj, "_orig_mod"):
            module.model = model_obj._orig_mod

    def _build_checkpoint_callback(self, *, early_stopping_cfg: dict[str, Any]) -> pl.callbacks.ModelCheckpoint:
        es_cfg_inner = dict(early_stopping_cfg.get("config") or {})
        monitor = str(es_cfg_inner.get("monitor", "val_loss"))
        mode = str(es_cfg_inner.get("mode", "min"))
        return pl.callbacks.ModelCheckpoint(
            dirpath=self._checkpoint_dir,
            filename="best",
            monitor=monitor,
            mode=mode,
            save_top_k=1,
            save_last=False,
            auto_insert_metric_name=False,
        )

    def fit(self, model, train_dataloaders=None, val_dataloaders=None, dataloaders=None, ckpt_path=None):  # type: ignore[override]
        try:
            fit_sig = inspect.signature(super().fit)
            fit_kwargs: dict[str, Any] = {
                "model": model,
                "train_dataloaders": train_dataloaders,
                "val_dataloaders": val_dataloaders,
                "dataloaders": dataloaders,
                "ckpt_path": ckpt_path,
            }
            supported_kwargs = {k: v for k, v in fit_kwargs.items() if k in fit_sig.parameters}
            out = super().fit(**supported_kwargs)
            best_model_path = str(getattr(self._checkpoint_callback, "best_model_path", "") or "")
            if best_model_path:
                best_ckpt = torch.load(best_model_path, map_location="cpu")
                state_dict = best_ckpt.get("state_dict", best_ckpt)
                model.load_state_dict(state_dict, strict=True)
            return out
        finally:
            if self._restore_eager_after_fit:
                self._restore_eager_model_if_compiled(model)
            shutil.rmtree(self._checkpoint_dir, ignore_errors=True)
