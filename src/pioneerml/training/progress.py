"""Custom progress bar for clean, informative training output."""

from __future__ import annotations

import sys
import time
from typing import Any, Optional

from pytorch_lightning.callbacks import Callback


class CleanProgressBar(Callback):
    """
    Clean, single-line progress bar for training and validation.

    Shows epoch progress, metrics, and ETA in a compact format without
    the messy multi-line TQDM output from PyTorch Lightning.

    Example output:
        Epoch 1/5 | Train: 100% [40/40] loss=0.123 | Val: 100% [10/10] loss=0.045 | ETA: 2m 15s
    """

    def __init__(self, bar_width: int = 30):
        super().__init__()
        self.bar_width = bar_width
        self._current_epoch = 0
        self._max_epochs = 0
        self._train_batch_idx = 0
        self._train_total_batches = 0
        self._val_batch_idx = 0
        self._val_total_batches = 0
        self._train_metrics = {}
        self._val_metrics = {}

        # Timing for ETA
        self._epoch_start_time = None
        self._training_start_time = None
        self._epoch_times = []  # Track time per epoch for better ETA
        self._batch_start_time = None
        self._batch_times = []  # Track time per batch for within-epoch ETA

    def on_train_start(self, trainer, pl_module) -> None:
        """Initialize training parameters."""
        self._max_epochs = trainer.max_epochs
        self._training_start_time = time.time()
        print(f"\nStarting training for {self._max_epochs} epochs...")
        print("=" * 80)

    def on_train_epoch_start(self, trainer, pl_module) -> None:
        """Reset counters at the start of each epoch."""
        self._current_epoch = trainer.current_epoch + 1
        self._epoch_start_time = time.time()
        self._batch_start_time = time.time()
        self._train_batch_idx = 0
        self._train_total_batches = trainer.num_training_batches
        self._train_metrics = {}
        self._val_metrics = {}
        self._batch_times = []  # Reset batch times for new epoch

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx) -> None:
        """Update progress after each training batch."""
        self._train_batch_idx = batch_idx + 1

        # Track batch time
        if self._batch_start_time is not None:
            batch_time = time.time() - self._batch_start_time
            self._batch_times.append(batch_time)
            self._batch_start_time = time.time()

        # Extract loss from outputs
        if outputs and "loss" in outputs:
            self._train_metrics["loss"] = float(outputs["loss"])

        self._update_display(mode="train")

    def on_validation_epoch_start(self, trainer, pl_module) -> None:
        """Reset validation counters."""
        self._val_batch_idx = 0
        self._val_total_batches = (
            trainer.num_val_batches[0] if trainer.num_val_batches else 0
        )

    def on_validation_batch_end(
        self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0
    ) -> None:
        """Update progress after each validation batch."""
        self._val_batch_idx = batch_idx + 1
        self._update_display(mode="val")

    def on_validation_end(self, trainer, pl_module) -> None:
        """Update metrics after validation completes."""
        # Get validation metrics from trainer
        if hasattr(trainer, "callback_metrics"):
            for key, value in trainer.callback_metrics.items():
                if "val" in key and hasattr(value, "item"):
                    metric_name = key.replace("val_", "")
                    self._val_metrics[metric_name] = float(value.item())

        # Track epoch time for ETA calculation
        if self._epoch_start_time is not None:
            epoch_duration = time.time() - self._epoch_start_time
            self._epoch_times.append(epoch_duration)

        self._update_display(mode="complete")
        print()  # New line after epoch completes

    def on_train_end(self, trainer, pl_module) -> None:
        """Print summary when training finishes."""
        if self._training_start_time is not None:
            total_time = time.time() - self._training_start_time
            print("=" * 80)
            print(f"Training complete! Total time: {self._format_time(total_time)}")
        else:
            print("=" * 80)
            print("Training complete!")

    def _update_display(self, mode: str = "train") -> None:
        """
        Update the progress display.

        Args:
            mode: Display mode - 'train', 'val', or 'complete'
        """
        # Build the display string
        parts = []

        # Epoch info
        parts.append(f"Epoch {self._current_epoch}/{self._max_epochs}")

        # Training progress
        if self._train_total_batches > 0:
            train_pct = (self._train_batch_idx / self._train_total_batches) * 100
            train_bar = self._make_bar(self._train_batch_idx, self._train_total_batches)
            train_metrics = self._format_metrics(self._train_metrics)
            parts.append(
                f"Train: {train_pct:5.1f}% {train_bar} [{self._train_batch_idx}/{self._train_total_batches}]{train_metrics}"
            )

        # Validation progress (only show if we're in validation)
        if mode in ("val", "complete") and self._val_total_batches > 0:
            val_pct = (self._val_batch_idx / self._val_total_batches) * 100
            val_bar = self._make_bar(self._val_batch_idx, self._val_total_batches)
            val_metrics = self._format_metrics(self._val_metrics)
            parts.append(
                f"Val: {val_pct:5.1f}% {val_bar} [{self._val_batch_idx}/{self._val_total_batches}]{val_metrics}"
            )

        # ETA - show during training and after epoch completes
        eta = self._calculate_eta(mode)
        if eta:
            parts.append(f"ETA: {eta}")

        # Print with carriage return to overwrite the line
        display = " | ".join(parts)
        sys.stdout.write(f"\r{display:<140}")
        sys.stdout.flush()

    def _make_bar(self, current: int, total: int) -> str:
        """Create a text progress bar."""
        if total == 0:
            return "[" + " " * self.bar_width + "]"

        filled = int((current / total) * self.bar_width)
        bar = "█" * filled + "░" * (self.bar_width - filled)
        return f"[{bar}]"

    def _format_metrics(self, metrics: dict[str, float]) -> str:
        """Format metrics for display."""
        if not metrics:
            return ""

        metric_strs = [f"{k}={v:.4f}" for k, v in metrics.items()]
        return " " + " ".join(metric_strs)

    def _calculate_eta(self, mode: str = "train") -> str | None:
        """Calculate estimated time to completion.

        Args:
            mode: Display mode - 'train' for within-epoch ETA, 'complete' for epoch-based ETA
        """
        if self._current_epoch >= self._max_epochs:
            return None

        eta_seconds = 0.0

        # Calculate remaining time in current epoch (during training)
        if mode == "train" and len(self._batch_times) > 0 and self._train_total_batches > 0:
            # Use recent batch times for estimate
            recent_batch_times = self._batch_times[-10:]  # Last 10 batches
            avg_batch_time = sum(recent_batch_times) / len(recent_batch_times)

            # Time left in current epoch (train + val)
            remaining_train_batches = self._train_total_batches - self._train_batch_idx
            current_epoch_eta = remaining_train_batches * avg_batch_time

            # Add validation time estimate if we have data from previous epochs
            if len(self._epoch_times) > 0:
                # Validation typically takes less time than training
                # Estimate ~20% of epoch time for validation
                avg_epoch_time = sum(self._epoch_times[-3:]) / len(self._epoch_times[-3:])
                val_time_estimate = avg_epoch_time * 0.2
                current_epoch_eta += val_time_estimate

            eta_seconds += current_epoch_eta

        # Calculate remaining epochs (for both modes, if we have epoch history)
        if len(self._epoch_times) > 0:
            recent_times = self._epoch_times[-3:]  # Last 3 epochs
            avg_epoch_time = sum(recent_times) / len(recent_times)
            remaining_epochs = self._max_epochs - self._current_epoch
            eta_seconds += avg_epoch_time * remaining_epochs

        return self._format_time(eta_seconds) if eta_seconds > 0 else None

    def _format_time(self, seconds: float) -> str:
        """Format time duration in human-readable format."""
        if seconds < 60:
            return f"{int(seconds)}s"
        elif seconds < 3600:
            minutes = int(seconds // 60)
            secs = int(seconds % 60)
            return f"{minutes}m {secs}s"
        else:
            hours = int(seconds // 3600)
            minutes = int((seconds % 3600) // 60)
            return f"{hours}h {minutes}m"
