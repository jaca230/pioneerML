from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class AngularUnitVectorLoss(nn.Module):
    """Directional loss: mean(1 - cos(pred_vec, target_vec)) on q50 vector."""

    def __init__(
        self,
        *,
        quantiles: tuple[float, ...] = (0.16, 0.50, 0.84),
        num_outputs: int = 3,
        unit_norm_weight: float = 0.05,
    ) -> None:
        super().__init__()
        self.quantiles = tuple(float(q) for q in quantiles)
        if 0.50 not in self.quantiles:
            raise ValueError("quantiles must include 0.50 for median-direction supervision.")
        self.num_outputs = int(num_outputs)
        if self.num_outputs <= 0:
            raise ValueError("num_outputs must be > 0.")
        self.unit_norm_weight = float(unit_norm_weight)
        if self.unit_norm_weight < 0.0:
            raise ValueError("unit_norm_weight must be >= 0.")
        self.mid_index = int(self.quantiles.index(0.50))

    def forward(self, preds: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if preds.dim() != 2 or target.dim() != 2:
            raise ValueError(
                f"Expected 2D tensors [B, D], got preds={tuple(preds.shape)}, target={tuple(target.shape)}"
            )

        # Support either flattened quantile layout [B, num_outputs*num_quantiles] or [B, num_outputs].
        num_q = len(self.quantiles)
        expected_qdim = int(self.num_outputs * num_q)
        if int(preds.shape[1]) == expected_qdim:
            p = preds.view(preds.shape[0], self.num_outputs, num_q)[:, :, self.mid_index]
        elif int(preds.shape[1]) == self.num_outputs:
            p = preds
        else:
            raise ValueError(
                f"Unexpected preds dim {int(preds.shape[1])}; expected {self.num_outputs} or {expected_qdim}."
            )

        if int(target.shape[1]) == expected_qdim:
            t = target.view(target.shape[0], self.num_outputs, num_q)[:, :, 0]
        elif int(target.shape[1]) == self.num_outputs:
            t = target
        else:
            raise ValueError(
                f"Unexpected target dim {int(target.shape[1])}; expected {self.num_outputs} or {expected_qdim}."
            )

        p_norm = torch.linalg.vector_norm(p, ord=2, dim=1).clamp_min(1e-8)
        p_unit = p / p_norm.unsqueeze(1)
        t_unit = F.normalize(t, p=2, dim=1, eps=1e-8)
        dot = torch.sum(p_unit * t_unit, dim=1).clamp(-1.0, 1.0)
        angular = (1.0 - dot).mean()
        if self.unit_norm_weight <= 0.0:
            return angular
        unit_reg = torch.mean((p_norm - 1.0) ** 2)
        return angular + (self.unit_norm_weight * unit_reg)


class QuantileAngularLoss(nn.Module):
    """Hybrid loss for direction quantiles.

    - Pinball loss on per-component quantiles (q16/q50/q84)
    - Angular loss on the q50 direction vector
    - Optional q50 norm regularization toward unit length
    """

    def __init__(
        self,
        *,
        quantiles: tuple[float, ...] = (0.16, 0.50, 0.84),
        num_outputs: int = 3,
        pinball_weight: float = 0.0,
        angular_weight: float = 1.0,
        unit_norm_weight: float = 0.0,
        normalize_target: bool = False,
        clamp_dot: bool = False,
    ) -> None:
        super().__init__()
        if not quantiles:
            raise ValueError("quantiles must be non-empty.")
        self.quantiles = tuple(float(q) for q in quantiles)
        if 0.50 not in self.quantiles:
            raise ValueError("quantiles must include 0.50.")
        self.num_outputs = int(num_outputs)
        if self.num_outputs <= 0:
            raise ValueError("num_outputs must be > 0.")
        self.pinball_weight = float(pinball_weight)
        self.angular_weight = float(angular_weight)
        self.unit_norm_weight = float(unit_norm_weight)
        self.normalize_target = bool(normalize_target)
        self.clamp_dot = bool(clamp_dot)
        if self.pinball_weight < 0.0 or self.angular_weight < 0.0 or self.unit_norm_weight < 0.0:
            raise ValueError("Loss weights must be >= 0.")

        self.mid_index = int(self.quantiles.index(0.50))
        q = torch.tensor(self.quantiles, dtype=torch.float32).view(1, 1, -1)
        self.register_buffer("qvals", q, persistent=False)

    def forward(self, preds: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if preds.dim() != 2 or target.dim() != 2:
            raise ValueError(
                f"Expected 2D tensors [B, D], got preds={tuple(preds.shape)}, target={tuple(target.shape)}"
            )
        num_q = len(self.quantiles)
        expected = int(self.num_outputs * num_q)
        if int(preds.shape[1]) != expected or int(target.shape[1]) != expected:
            raise ValueError(
                f"Expected last dimension {expected} (=num_outputs*num_quantiles), "
                f"got preds={int(preds.shape[1])}, target={int(target.shape[1])}"
            )

        p = preds.view(preds.shape[0], self.num_outputs, num_q)
        t = target.view(target.shape[0], self.num_outputs, num_q)
        t_base = t[:, :, :1]

        q = self.qvals.to(dtype=p.dtype, device=p.device)
        err = t_base - p
        pinball = torch.maximum(q * err, (q - 1.0) * err).mean()

        # Omar-style angle loss on median direction:
        # pred_unit = normalize(pred_q50), loss = mean(1 - dot(pred_unit, target))
        p50 = p[:, :, self.mid_index]
        t_dir = t_base[:, :, 0]
        p50_norm = torch.linalg.vector_norm(p50, ord=2, dim=1).clamp_min(1e-8)
        p50_unit = p50 / p50_norm.unsqueeze(1)
        if self.normalize_target:
            t_dir = F.normalize(t_dir, p=2, dim=1, eps=1e-8)
        dot = torch.sum(p50_unit * t_dir, dim=1)
        if self.clamp_dot:
            dot = dot.clamp(-1.0, 1.0)
        angular = (1.0 - dot).mean()

        if self.unit_norm_weight > 0.0:
            unit_reg = torch.mean((p50_norm - 1.0) ** 2)
        else:
            unit_reg = angular.new_zeros(())

        return (
            self.pinball_weight * pinball
            + self.angular_weight * angular
            + self.unit_norm_weight * unit_reg
        )
