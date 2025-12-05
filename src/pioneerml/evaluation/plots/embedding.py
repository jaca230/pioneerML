from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.manifold import TSNE

from .base import BasePlot, _prepare_classification_inputs, _resolve_labels, _to_numpy

try:
    from IPython.display import display  # type: ignore
except Exception:  # pragma: no cover
    display = None


class EmbeddingSpacePlot(BasePlot):
    name = "embedding_space"

    def render(
        self,
        embeddings: Any,
        targets: Any,
        *,
        class_names: Sequence[str] | None = None,
        method: str = "tsne",
        perplexity: float = 30.0,
        n_components: int = 2,
        save_path: str | Path | None = None,
        show: bool = False,
    ) -> str | None:
        emb = _to_numpy(embeddings)
        tgt = _to_numpy(targets)

        if emb.ndim != 2:
            raise ValueError(f"Embeddings must be 2D [N, D]; got shape {emb.shape}")

        # Handle flattened labels
        if tgt.ndim == 1 and emb.shape[0] > 0 and tgt.size % emb.shape[0] == 0:
            tgt = tgt.reshape(emb.shape[0], -1)

        if tgt.shape[0] != emb.shape[0]:
            raise ValueError(f"Embeddings and targets must align on batch dimension: {emb.shape[0]} vs {tgt.shape[0]}")

        # Collapse targets to single class per sample for coloring
        if tgt.ndim > 1 and tgt.shape[1] > 1:
            tgt_idx = np.argmax(tgt, axis=1)
            labels = _resolve_labels(tgt.shape[1], class_names)
        else:
            tgt_idx = tgt.reshape(-1).astype(int)
            max_cls = int(tgt_idx.max()) + 1 if tgt_idx.size else 0
            labels = _resolve_labels(max_cls, class_names)

        if method.lower() != "tsne":
            raise ValueError(f"Unsupported embedding projection method: {method}")

        reducer = TSNE(n_components=n_components, perplexity=perplexity, init="random", learning_rate="auto")
        emb_2d = reducer.fit_transform(emb)

        fig, ax = plt.subplots(figsize=(6, 5))
        palette = sns.color_palette(n_colors=max(1, len(labels)))
        for cls in np.unique(tgt_idx):
            cls_mask = tgt_idx == cls
            label = labels[cls] if cls < len(labels) else str(cls)
            ax.scatter(
                emb_2d[cls_mask, 0],
                emb_2d[cls_mask, 1],
                s=12,
                alpha=0.35,
                label=label,
                color=palette[cls % len(palette)],
                edgecolors="none",
            )

        ax.set_title("Embedding Space (t-SNE)")
        ax.set_xlabel("Component 1")
        ax.set_ylabel("Component 2")
        ax.legend(markerscale=2, fontsize="small", frameon=True)
        ax.grid(True, linestyle="--", alpha=0.3)
        plt.tight_layout()

        if save_path is not None:
            save_path = str(save_path)
            fig.savefig(save_path, dpi=150, bbox_inches="tight")
        if show:
            backend = plt.get_backend().lower()
            if backend.startswith("agg"):
                if display is not None:
                    try:
                        display(fig)
                    except Exception:
                        pass
            else:
                plt.show()
        plt.close(fig)
        return save_path
