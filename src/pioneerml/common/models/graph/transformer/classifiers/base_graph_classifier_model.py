from __future__ import annotations

import torch

from pioneerml.common.models.graph.transformer.base_graph_transformer_model import BaseGraphTransformerModel


class BaseGraphClassifierModel(BaseGraphTransformerModel):
    """Base class for graph classifier models."""

    @torch.jit.ignore
    def extract_embeddings(self, data) -> torch.Tensor:
        raise NotImplementedError(f"{self.__class__.__name__} does not implement extract_embeddings.")
