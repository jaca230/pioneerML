from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

import os
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.neighbors import NearestNeighbors

try:
    from tqdm import tqdm
except ImportError:
    tqdm = None

from .base import BasePlot, _prepare_classification_inputs, _resolve_labels, _to_numpy

# Disable numba JIT for UMAP globally to avoid environment-specific compile issues
os.environ.setdefault("NUMBA_DISABLE_JIT", "1")
# Also toggle via numba API in case the environment variable was too late
try:
    import numba  # type: ignore

    numba.config.DISABLE_JIT = True
except Exception:  # pragma: no cover
    numba = None
try:
    import umap  # type: ignore
except Exception:  # pragma: no cover
    umap = None

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
        random_state: int | None = None,
        title: str | None = None,
        max_samples: int | None = None,
        pre_pca_components: int | None = None,
        precompute_neighbors: bool = True,
        verbose: bool = False,
        save_path: str | Path | None = None,
        show: bool = False,
    ) -> str | None:
        emb_raw = _to_numpy(embeddings)
        if isinstance(emb_raw, list):
            emb_list = [np.asarray(x) for x in emb_raw if x is not None and len(np.asarray(x)) > 0]
            if not emb_list:
                raise ValueError("Embeddings list is empty after filtering.")
            emb = np.vstack(emb_list)
        else:
            emb = np.asarray(emb_raw)
        if emb.ndim != 2 or emb.shape[0] == 0 or emb.shape[1] == 0:
            raise ValueError(f"Embeddings must be 2D (N, D) with nonzero shape; got {emb.shape}")
        if not np.isfinite(emb).all():
            raise ValueError("Embeddings contain NaN or inf values.")
        emb = emb.astype(np.float32, copy=False)

        tgt = _to_numpy(targets)

        # Optional subsample for speed
        if max_samples is not None and emb.shape[0] > max_samples:
            rng = np.random.default_rng(random_state)
            idx = rng.choice(emb.shape[0], size=max_samples, replace=False)
            emb = emb[idx]
            tgt = tgt[idx]
            if verbose:
                print(f"[embedding] Subsampled to {emb.shape[0]} samples for speed.")

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

        method_lower = method.lower()
        emb_2d = None
        emb_used = emb

        # Optional PCA pre-reduction before heavier methods
        if pre_pca_components is not None and method_lower in {"umap", "tsne"}:
            pca_pre = PCA(n_components=min(pre_pca_components, emb.shape[1]), random_state=random_state)
            emb_used = pca_pre.fit_transform(emb)
            if verbose:
                print(f"[embedding] Pre-reduced to {emb_used.shape[1]} dims via PCA.")

        if method_lower == "pca":
            reducer = PCA(n_components=n_components, random_state=random_state)
            emb_2d = reducer.fit_transform(emb_used)
        elif method_lower == "umap":
            if umap is None:
                raise ImportError("umap-learn is required for UMAP embeddings. Install via `pip install umap-learn`.")
            if emb_used.shape[0] < 3:
                raise ValueError("UMAP requires at least 3 samples.")

            # Filter out any rows with NaN, inf, or all zeros
            valid_mask = np.isfinite(emb_used).all(axis=1) & (emb_used.std(axis=1) > 1e-8)
            if valid_mask.sum() < 3:
                raise ValueError("Not enough valid embedding rows after filtering NaN/inf/constant rows.")

            emb_clean = emb_used[valid_mask]
            tgt_clean = tgt_idx[valid_mask]

            # Calculate n_neighbors more carefully
            n_samples = emb_clean.shape[0]
            n_neighbors = min(15, max(2, n_samples - 1))

            try:
                # Precompute neighbors with sklearn to avoid pynndescent/numba path
                knn_indices = None
                knn_dists = None
                if precompute_neighbors:
                    if verbose:
                        print(f"[embedding] Computing {n_neighbors}-NN graph with sklearn for UMAP...")
                    nn = NearestNeighbors(
                        n_neighbors=n_neighbors,
                        metric="euclidean",
                        algorithm="auto",
                        n_jobs=1,
                    )
                    nn.fit(emb_clean)
                    knn_dists, knn_indices = nn.kneighbors(emb_clean, return_distance=True)
                    knn_dists = knn_dists.astype(np.float32, copy=False)

                reducer = umap.UMAP(
                    n_components=n_components,
                    random_state=random_state,
                    n_neighbors=n_neighbors,
                    metric="euclidean",
                    n_jobs=1,
                    low_memory=True,
                    init="spectral",
                    verbose=False,  # Disable UMAP's verbose to avoid conflicts with tqdm
                    force_approximation_algorithm=precompute_neighbors,
                    precomputed_knn=(knn_indices, knn_dists, None),
                )
                
                # Show progress bar for UMAP (it's a slow operation)
                if tqdm is not None:
                    import sys
                    import time
                    import threading
                    
                    # Simple tqdm progress bar that updates to show elapsed time
                    pbar = tqdm(
                        total=100,
                        desc=f"UMAP ({emb_clean.shape[0]} samples)",
                        file=sys.stderr,
                        bar_format="{desc}: {percentage:3.0f}%|{bar}| {elapsed}",
                    )
                    
                    stop_updating = threading.Event()
                    
                    def update_progress():
                        """Update progress bar every second to show elapsed time."""
                        while not stop_updating.is_set():
                            time.sleep(1)
                            if not stop_updating.is_set() and pbar.n < 99:
                                pbar.update(1)
                    
                    update_thread = threading.Thread(target=update_progress, daemon=True)
                    update_thread.start()
                    
                    try:
                        emb_2d = reducer.fit_transform(emb_clean)
                        # Complete the bar
                        while pbar.n < 100:
                            pbar.update(100 - pbar.n)
                    finally:
                        stop_updating.set()
                        pbar.close()
                else:
                    if verbose:
                        print(f"[embedding] Running UMAP on {emb_clean.shape[0]} samples...")
                    emb_2d = reducer.fit_transform(emb_clean)
                # Update tgt_idx to use cleaned version
                tgt_idx = tgt_clean
            except Exception as e:
                raise ValueError(
                    f"UMAP failed: embeddings must be a 2D (N,D) float array with no empty rows. "
                    f"Got shape {emb.shape}. Original error: {e}"
                ) from e
        elif method_lower == "tsne":
            reducer = TSNE(
                n_components=n_components,
                perplexity=perplexity,
                init="random",
                learning_rate="auto",
                random_state=random_state,
            )
            emb_2d = reducer.fit_transform(emb_used)
        else:
            raise ValueError(f"Unsupported embedding projection method: {method}")
        
        if emb_2d is None:
            raise RuntimeError("Failed to compute 2D embedding projection")

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

        plot_title = title or f"Embedding Space ({method.upper()})"
        ax.set_title(plot_title)
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
