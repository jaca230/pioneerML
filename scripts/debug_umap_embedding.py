"""
Quick harness to reproduce embedding plot calls (PCA / t-SNE / UMAP) with dummy data.

Run:
    python scripts/debug_umap_embedding.py
"""

from __future__ import annotations

import pathlib
import numpy as np

from pioneerml.evaluation.plots import plot_embedding_space


def main():
    # Smaller shapes for quick iteration; bump as needed.
    n_samples = 500
    n_dims = 64
    n_classes = 3

    rng = np.random.default_rng(42)
    embeddings = rng.normal(size=(n_samples, n_dims)).astype(np.float32)

    # One-hot-ish targets; here we use class indices then one-hot encode.
    target_idx = rng.integers(low=0, high=n_classes, size=n_samples)
    targets = np.zeros((n_samples, n_classes), dtype=np.float32)
    targets[np.arange(n_samples), target_idx] = 1.0
    class_names = [f"class_{i}" for i in range(n_classes)]

    out_dir = pathlib.Path("artifacts/debug_umap")
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Plotting t-SNE...")
    plot_embedding_space(
        embeddings=embeddings,
        targets=targets,
        class_names=class_names,
        method="tsne",
        title="Debug t-SNE",
        perplexity=30.0,
        n_components=2,
        random_state=42,
        save_path=out_dir / "tsne.png",
        show=False,
    )

    print("Plotting PCA...")
    plot_embedding_space(
        embeddings=embeddings,
        targets=targets,
        class_names=class_names,
        method="pca",
        title="Debug PCA",
        n_components=2,
        random_state=42,
        save_path=out_dir / "pca.png",
        show=False,
    )

    print("Plotting UMAP...")
    plot_embedding_space(
        embeddings=embeddings,
        targets=targets,
        class_names=class_names,
        method="umap",
        title="Debug UMAP",
        n_components=2,
        random_state=42,
        save_path=out_dir / "umap.png",
        show=False,
    )

    print(f"Saved plots to {out_dir}")


if __name__ == "__main__":
    main()
