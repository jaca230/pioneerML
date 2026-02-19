from __future__ import annotations

import torch
import torch.nn as nn


class QuantilePinballLoss(nn.Module):
    """Pinball loss for flattened `[output_i x quantile_j]` regression outputs."""

    def __init__(self, *, quantiles: tuple[float, ...] = (0.16, 0.50, 0.84), num_outputs: int = 3) -> None:
        super().__init__()
        if not quantiles:
            raise ValueError("quantiles must be non-empty.")
        self.quantiles = tuple(float(q) for q in quantiles)
        self.num_outputs = int(num_outputs)
        if self.num_outputs <= 0:
            raise ValueError("num_outputs must be > 0.")
        q = torch.tensor(self.quantiles, dtype=torch.float32).view(1, 1, -1)
        self.register_buffer("qvals", q, persistent=False)

    def forward(self, preds: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if preds.dim() != 2:
            raise ValueError(f"Expected preds to be 2D [B, D], got shape {tuple(preds.shape)}")
        if target.dim() != 2:
            raise ValueError(f"Expected target to be 2D [B, D], got shape {tuple(target.shape)}")
        num_q = len(self.quantiles)
        expected = self.num_outputs * num_q
        if int(preds.shape[1]) != expected or int(target.shape[1]) != expected:
            raise ValueError(
                f"Expected last dimension {expected} (=num_outputs*num_quantiles), "
                f"got preds={int(preds.shape[1])}, target={int(target.shape[1])}"
            )
        p = preds.view(preds.shape[0], self.num_outputs, num_q)
        t = target.view(target.shape[0], self.num_outputs, num_q)
        # Targets are repeated across quantile axis in our loaders; use the first slice as canonical truth.
        t_base = t[:, :, :1]
        err = t_base - p
        q = self.qvals.to(dtype=p.dtype, device=p.device)
        loss = torch.maximum(q * err, (q - 1.0) * err)
        return loss.mean()

