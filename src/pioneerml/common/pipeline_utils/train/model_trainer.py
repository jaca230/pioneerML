from __future__ import annotations

from typing import Any, Callable

import pytorch_lightning as pl
from pioneerml.common.zenml.utils import detect_available_accelerator


class ModelTrainer:
    """Shared orchestration for module fitting with Lightning."""

    def fit(
        self,
        *,
        module,
        graphs: list,
        max_epochs: int,
        batch_size: int,
        shuffle: bool,
        grad_clip: float | None,
        trainer_kwargs: dict | None,
        loader_cls,
        collate_fn: Callable | None = None,
    ) -> Any:
        if not graphs:
            raise RuntimeError("No non-empty graphs found in dataset for training.")

        loader_kwargs: dict[str, Any] = {
            "batch_size": int(batch_size),
        }
        if collate_fn is not None:
            loader_kwargs["collate_fn"] = collate_fn

        train_loader = loader_cls(graphs, shuffle=bool(shuffle), **loader_kwargs)
        val_loader = loader_cls(graphs, shuffle=False, **loader_kwargs)

        merged_trainer_kwargs = dict(trainer_kwargs or {})
        if grad_clip is not None:
            merged_trainer_kwargs.setdefault("gradient_clip_val", float(grad_clip))

        accelerator, devices = detect_available_accelerator()
        trainer = pl.Trainer(
            accelerator=accelerator,
            devices=devices,
            max_epochs=int(max_epochs),
            enable_checkpointing=False,
            logger=False,
            **merged_trainer_kwargs,
        )
        trainer.fit(module, train_dataloaders=train_loader, val_dataloaders=val_loader)
        return module
