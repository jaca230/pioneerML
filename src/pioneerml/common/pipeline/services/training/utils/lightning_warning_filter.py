import warnings

try:
    from lightning_fabric.utilities.rank_zero import LightningDeprecationWarning
except Exception:  # pragma: no cover
    LightningDeprecationWarning = None


class LightningWarningFilter:
    def apply_default(self) -> None:
        warnings.filterwarnings("ignore", message="isinstance\\(treespec, LeafSpec\\) is deprecated.*")
        warnings.filterwarnings(
            "ignore",
            message="`isinstance\\(treespec, LeafSpec\\)` is deprecated.*",
            category=FutureWarning,
        )
        if LightningDeprecationWarning is not None:
            warnings.filterwarnings("ignore", category=LightningDeprecationWarning)

