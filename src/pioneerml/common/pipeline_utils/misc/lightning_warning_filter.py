from __future__ import annotations

import warnings


class LightningWarningFilter:
    """Apply warning filters used by Lightning-based training steps."""

    def apply_default(self) -> None:
        warnings.filterwarnings(
            "ignore",
            message="The 'train_dataloader' does not have many workers.*",
            category=UserWarning,
        )
        warnings.filterwarnings(
            "ignore",
            message="The 'val_dataloader' does not have many workers.*",
            category=UserWarning,
        )
        warnings.filterwarnings(
            "ignore",
            message="isinstance\\(treespec, LeafSpec\\) is deprecated.*",
            category=DeprecationWarning,
        )
        warnings.filterwarnings(
            "ignore",
            message="isinstance\\(treespec, LeafSpec\\) is deprecated.*",
            category=UserWarning,
        )
        warnings.filterwarnings(
            "ignore",
            message="isinstance\\(treespec, LeafSpec\\) is deprecated.*",
            category=Warning,
            module="pytorch_lightning\\.utilities\\._pytree",
        )
