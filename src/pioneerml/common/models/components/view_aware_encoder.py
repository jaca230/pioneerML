"""Stereo-aware hit encoder."""

from __future__ import annotations

import torch
import torch.nn as nn


class ViewAwareEncoder(nn.Module):
    def __init__(self, prob_dim, hidden_dim):
        super().__init__()
        self.prob_dim = prob_dim
        self.view_x_val = 0
        self.view_y_val = 1
        self.feature_proj = nn.Linear(3 + prob_dim, hidden_dim)
        self.view_embedding = nn.Embedding(2, hidden_dim)
        nn.init.normal_(self.view_embedding.weight, std=0.02)

    def forward(self, x: torch.Tensor, probs: torch.Tensor | None = None) -> torch.Tensor:
        phys_feats = x[:, :3]
        raw_view = x[:, 3].long()

        embedding_idx = torch.zeros_like(raw_view)
        embedding_idx[raw_view == self.view_y_val] = 1

        if probs is None:
            probs = torch.zeros(x.size(0), self.prob_dim, device=x.device)

        features = torch.cat([phys_feats, probs], dim=1)
        hit_embed = self.feature_proj(features)

        return hit_embed + self.view_embedding(embedding_idx)


__all__ = ["ViewAwareEncoder"]
