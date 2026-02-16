"""Stereo-aware group classifier."""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn
from torch_geometric.nn import AttentionalAggregation, JumpingKnowledge

from pioneerml.common.models.base import GraphModel
from pioneerml.common.models.blocks import FullGraphTransformerBlock
from pioneerml.common.models.components.view_aware_encoder import ViewAwareEncoder


class GroupClassifierStereo(GraphModel):
    def __init__(
        self,
        in_dim: int = 4,
        edge_dim: int = 4,
        hidden: int = 200,
        heads: int = 4,
        num_blocks: int = 2,
        dropout: float = 0.1,
        num_classes: int = 3,
    ):
        super().__init__(in_channels=in_dim, hidden=hidden, edge_dim=edge_dim, dropout=dropout)

        self.input_embed = ViewAwareEncoder(prob_dim=0, hidden_dim=hidden)
        self.view_x_val = int(self.input_embed.view_x_val)
        self.view_y_val = int(self.input_embed.view_y_val)

        self.blocks = nn.ModuleList(
            [
                FullGraphTransformerBlock(
                    hidden=hidden,
                    heads=heads,
                    edge_dim=edge_dim,
                    dropout=dropout,
                )
                for _ in range(num_blocks)
            ]
        )

        self.jk = JumpingKnowledge(mode="cat")
        jk_dim = hidden * num_blocks

        self.pool_x = AttentionalAggregation(
            nn.Sequential(nn.Linear(jk_dim, jk_dim // 2), nn.ReLU(), nn.Linear(jk_dim // 2, 1))
        )
        self.pool_y = AttentionalAggregation(
            nn.Sequential(nn.Linear(jk_dim, jk_dim // 2), nn.ReLU(), nn.Linear(jk_dim // 2, 1))
        )

        concat_dim = (jk_dim * 2) + 2  # pools + valid bits

        self.head = nn.Sequential(
            nn.Linear(concat_dim, jk_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(jk_dim, jk_dim // 2),
            nn.ReLU(),
            nn.Linear(jk_dim // 2, num_classes),
        )

    @torch.jit.ignore
    def forward(self, data) -> torch.Tensor:
        num_graphs = int(getattr(data, "num_graphs", 0))
        return self.forward_tensors(data.x, data.edge_index, data.edge_attr, data.batch, num_graphs=num_graphs)

    def forward_tensors(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
        batch: torch.Tensor,
        num_graphs: int | None = None,
    ) -> torch.Tensor:
        x_embed = self.input_embed(x)

        xs = []
        for block in self.blocks:
            x_embed = block(x_embed, edge_index, edge_attr)
            xs.append(x_embed)
        x_cat = self.jk(xs)

        raw_view = x[:, 3].to(torch.long)
        mask_x = raw_view == self.view_x_val
        mask_y = raw_view == self.view_y_val

        if num_graphs is None:
            num_graphs = int(torch.bincount(batch).shape[0]) if batch.numel() > 0 else 0
        if int(num_graphs) == 0:
            return x_cat.new_zeros((0, self.head[-1].out_features))

        pooled_x = self.pool_x(x_cat[mask_x], batch[mask_x], dim_size=int(num_graphs))
        pooled_y = self.pool_y(x_cat[mask_y], batch[mask_y], dim_size=int(num_graphs))
        counts_x = torch.bincount(batch[mask_x], minlength=int(num_graphs)).to(x_cat.dtype)
        counts_y = torch.bincount(batch[mask_y], minlength=int(num_graphs)).to(x_cat.dtype)
        has_x = (counts_x > 0).to(x_cat.dtype).unsqueeze(1)
        has_y = (counts_y > 0).to(x_cat.dtype).unsqueeze(1)

        out = torch.cat([pooled_x, pooled_y, has_x, has_y], dim=1)
        return self.head(out)

    @torch.jit.ignore
    def extract_embeddings(self, data) -> torch.Tensor:
        x_embed = self.input_embed(data.x)

        xs = []
        for block in self.blocks:
            x_embed = block(x_embed, data.edge_index, data.edge_attr)
            xs.append(x_embed)
        x_cat = self.jk(xs)

        raw_view = data.x[:, 3].to(torch.long)
        mask_x = raw_view == self.view_x_val
        mask_y = raw_view == self.view_y_val

        num_graphs = int(getattr(data, "num_graphs", 0))
        if num_graphs == 0:
            return x_cat.new_zeros((0, (x_cat.size(1) * 2) + 2))
        pooled_x = self.pool_x(x_cat[mask_x], data.batch[mask_x], dim_size=num_graphs)
        pooled_y = self.pool_y(x_cat[mask_y], data.batch[mask_y], dim_size=num_graphs)
        counts_x = torch.bincount(data.batch[mask_x], minlength=num_graphs).to(x_cat.dtype)
        counts_y = torch.bincount(data.batch[mask_y], minlength=num_graphs).to(x_cat.dtype)
        has_x = (counts_x > 0).to(x_cat.dtype).unsqueeze(1)
        has_y = (counts_y > 0).to(x_cat.dtype).unsqueeze(1)

        return torch.cat([pooled_x, pooled_y, has_x, has_y], dim=1)

    def export_torchscript(
        self,
        path: str | Path | None,
        *,
        prefer_cuda: bool = True,
        strict: bool = False,
    ) -> torch.jit.ScriptModule:
        device = torch.device("cuda") if prefer_cuda and torch.cuda.is_available() else torch.device("cpu")
        self.eval()
        self.to(device)

        class _Scriptable(nn.Module):
            def __init__(self, model: GroupClassifierStereo):
                super().__init__()
                self.model = model

            def forward(
                self,
                x: torch.Tensor,
                edge_index: torch.Tensor,
                edge_attr: torch.Tensor,
                batch: torch.Tensor,
            ) -> torch.Tensor:
                return self.model.forward_tensors(x, edge_index, edge_attr, batch)

        scriptable = _Scriptable(self)
        scripted = torch.jit.script(scriptable)
        if path is not None:
            scripted.save(str(path))
        return scripted


GroupClassifier = GroupClassifierStereo
