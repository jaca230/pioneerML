from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from .base import BasePlot, _to_numpy

try:
    from IPython.display import display  # type: ignore
except Exception:  # pragma: no cover
    display = None


class RegressionDiagnosticsPlot(BasePlot):
    name = "regression_diagnostics"

    def render(
        self,
        predictions: Any,
        targets: Any,
        *,
        save_path: str | Path | None = None,
        show: bool = False,
        component_names: list[str] | None = None,
    ) -> str | None:
        y_true = _to_numpy(targets)
        y_pred = _to_numpy(predictions)
        
        # Handle multi-dimensional outputs (e.g., [N, 2] for angles or [N, 3] for positions)
        if y_true.ndim > 1 and y_true.shape[1] > 1:
            num_components = y_true.shape[1]
            if component_names is None:
                component_names = [f"Component {i}" for i in range(num_components)]
            
            # Create subplots for each component
            fig, axes = plt.subplots(num_components, 2, figsize=(12, 4 * num_components))
            if num_components == 1:
                axes = axes.reshape(1, -1)
            
            for i in range(num_components):
                y_true_comp = y_true[:, i]
                y_pred_comp = y_pred[:, i]
                residuals = y_pred_comp - y_true_comp
                
                # Scatter plot
                axes[i, 0].scatter(y_true_comp, y_pred_comp, alpha=0.6)
                min_val = min(y_true_comp.min(), y_pred_comp.min())
                max_val = max(y_true_comp.max(), y_pred_comp.max())
                axes[i, 0].plot([min_val, max_val], [min_val, max_val], "k--", alpha=0.6)
                axes[i, 0].set_xlabel("True")
                axes[i, 0].set_ylabel("Predicted")
                axes[i, 0].set_title(f"{component_names[i]} - Predicted vs True")
                axes[i, 0].grid(True, linestyle="--", alpha=0.4)
                
                # Residuals
                sns.histplot(residuals, kde=True, ax=axes[i, 1])
                axes[i, 1].axvline(0, color="k", linestyle="--", alpha=0.7)
                axes[i, 1].set_xlabel("Residual")
                axes[i, 1].set_title(f"{component_names[i]} - Residual Distribution")
        else:
            # Single-dimensional output (flatten if needed)
            y_true = y_true.reshape(-1)
            y_pred = y_pred.reshape(-1)
            residuals = y_pred - y_true

            fig, axes = plt.subplots(1, 2, figsize=(10, 4))

            # Scatter plot
            axes[0].scatter(y_true, y_pred, alpha=0.6)
            min_val = min(y_true.min(), y_pred.min())
            max_val = max(y_true.max(), y_pred.max())
            axes[0].plot([min_val, max_val], [min_val, max_val], "k--", alpha=0.6)
            axes[0].set_xlabel("True")
            axes[0].set_ylabel("Predicted")
            axes[0].set_title("Predicted vs True")
            axes[0].grid(True, linestyle="--", alpha=0.4)

            # Residuals
            sns.histplot(residuals, kde=True, ax=axes[1])
            axes[1].axvline(0, color="k", linestyle="--", alpha=0.7)
            axes[1].set_xlabel("Residual")
            axes[1].set_title("Residual Distribution")

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


class EuclideanErrorHistogramPlot(BasePlot):
    name = "euclidean_error_histogram"

    def render(
        self,
        predictions: Any,
        targets: Any,
        *,
        log_scale: bool = False,
        bins: int | str = "auto",
        title: str | None = None,
        save_path: str | Path | None = None,
        show: bool = False,
    ) -> str | None:
        """
        Plot histogram of euclidean errors between predictions and targets.
        
        Args:
            predictions: Predicted values [N, D] where D is the dimensionality
            targets: True values [N, D]
            log_scale: If True, use log scale for y-axis
            bins: Number of bins or 'auto' for automatic binning
            title: Optional plot title
            save_path: Optional path to save the figure
            show: Whether to display the figure
        """
        y_true = _to_numpy(targets)
        y_pred = _to_numpy(predictions)
        
        # Ensure 2D arrays
        if y_true.ndim == 1:
            y_true = y_true.reshape(-1, 1)
        if y_pred.ndim == 1:
            y_pred = y_pred.reshape(-1, 1)
        
        if y_true.shape != y_pred.shape:
            raise ValueError(f"Predictions and targets must have the same shape. Got {y_pred.shape} vs {y_true.shape}")
        
        # Compute euclidean error for each sample
        errors = np.linalg.norm(y_pred - y_true, axis=1)
        
        fig, ax = plt.subplots(figsize=(8, 5))
        
        # Create histogram
        counts, bin_edges, patches = ax.hist(
            errors,
            bins=bins,
            edgecolor="black",
            alpha=0.7,
            color="steelblue",
        )
        
        # Add statistics
        mean_error = np.mean(errors)
        median_error = np.median(errors)
        std_error = np.std(errors)
        
        ax.axvline(mean_error, color="red", linestyle="--", linewidth=2, label=f"Mean: {mean_error:.4f}")
        ax.axvline(median_error, color="orange", linestyle="--", linewidth=2, label=f"Median: {median_error:.4f}")
        
        ax.set_xlabel("Euclidean Error", fontsize=12)
        ax.set_ylabel("Frequency" if not log_scale else "Frequency (log scale)", fontsize=12)
        
        if log_scale:
            ax.set_yscale("log")
        
        plot_title = title or "Euclidean Error Distribution"
        ax.set_title(plot_title, fontsize=14)
        ax.legend(fontsize=10)
        ax.grid(True, linestyle="--", alpha=0.3)
        
        # Add text box with statistics
        stats_text = f"Mean: {mean_error:.4f}\nMedian: {median_error:.4f}\nStd: {std_error:.4f}"
        ax.text(
            0.98,
            0.98,
            stats_text,
            transform=ax.transAxes,
            fontsize=9,
            verticalalignment="top",
            horizontalalignment="right",
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
        )
        
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


class ErrorEmbeddingSpacePlot(BasePlot):
    name = "error_embedding_space"

    def render(
        self,
        embeddings: Any,
        predictions: Any,
        targets: Any,
        *,
        method: str = "tsne",
        perplexity: float = 30.0,
        n_components: int = 2,
        random_state: int | None = None,
        title: str | None = None,
        max_samples: int | None = None,
        save_path: str | Path | None = None,
        show: bool = False,
    ) -> str | None:
        """
        Visualize embedding space colored by euclidean error magnitude.
        Useful for detecting overfitting patterns (e.g., clusters of high error).
        
        Args:
            embeddings: Model embeddings [N, D] to visualize
            predictions: Predicted values [N, D]
            targets: True values [N, D]
            method: Embedding method ('tsne', 'umap', or 'pca')
            perplexity: Perplexity for t-SNE
            n_components: Number of components for dimensionality reduction
            random_state: Random seed
            title: Optional plot title
            max_samples: Maximum number of samples to use (for speed)
            save_path: Optional path to save the figure
            show: Whether to display the figure
        """
        from .embedding import EmbeddingSpacePlot
        
        emb = _to_numpy(embeddings)
        y_pred = _to_numpy(predictions)
        y_true = _to_numpy(targets)
        
        # Ensure 2D arrays
        if emb.ndim == 1:
            emb = emb.reshape(-1, 1)
        if y_true.ndim == 1:
            y_true = y_true.reshape(-1, 1)
        if y_pred.ndim == 1:
            y_pred = y_pred.reshape(-1, 1)
        
        if emb.shape[0] != y_pred.shape[0] or emb.shape[0] != y_true.shape[0]:
            raise ValueError(
                f"Embeddings, predictions, and targets must have the same number of samples. "
                f"Got {emb.shape[0]}, {y_pred.shape[0]}, {y_true.shape[0]}"
            )
        
        # Compute euclidean errors
        errors = np.linalg.norm(y_pred - y_true, axis=1)
        
        # Use error magnitude as "class" labels for coloring
        # Bin errors into quantiles for better visualization
        n_bins = 5
        error_bins = np.quantile(errors, np.linspace(0, 1, n_bins + 1))
        error_labels = np.digitize(errors, error_bins) - 1
        error_labels = np.clip(error_labels, 0, n_bins - 1)
        
        # Create class names based on error ranges
        class_names = []
        for i in range(n_bins):
            low = error_bins[i]
            high = error_bins[i + 1] if i < n_bins - 1 else errors.max()
            class_names.append(f"Error: [{low:.3f}, {high:.3f}]")
        
        # Convert to one-hot for embedding plot
        targets_onehot = np.zeros((len(error_labels), n_bins), dtype=np.float32)
        targets_onehot[np.arange(len(error_labels)), error_labels] = 1.0
        
        # Use the existing embedding space plot
        embedding_plot = EmbeddingSpacePlot()
        return embedding_plot.render(
            embeddings=emb,
            targets=targets_onehot,
            class_names=class_names,
            method=method,
            perplexity=perplexity,
            n_components=n_components,
            random_state=random_state,
            title=title or f"Embedding Space Colored by Error ({method.upper()})",
            max_samples=max_samples,
            save_path=save_path,
            show=show,
        )
