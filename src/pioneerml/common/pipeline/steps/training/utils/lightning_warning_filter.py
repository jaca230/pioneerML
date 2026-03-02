import logging
import warnings

try:
    from lightning_fabric.utilities.rank_zero import LightningDeprecationWarning
except Exception:  # pragma: no cover
    LightningDeprecationWarning = None


class LightningWarningFilter:
    def apply_default(self) -> None:
        class _DropLitloggerTip(logging.Filter):
            def filter(self, record: logging.LogRecord) -> bool:
                msg = record.getMessage()
                return "For seamless cloud logging and experiment tracking, try installing [litlogger]" not in msg

        tip_filter = _DropLitloggerTip()
        for logger_name in (
            "lightning.pytorch.utilities.rank_zero",
            "pytorch_lightning.utilities.rank_zero",
        ):
            logging.getLogger(logger_name).addFilter(tip_filter)

        warnings.filterwarnings("ignore", message="isinstance\\(treespec, LeafSpec\\) is deprecated.*")
        warnings.filterwarnings(
            "ignore",
            message="`isinstance\\(treespec, LeafSpec\\)` is deprecated.*",
            category=FutureWarning,
        )
        warnings.filterwarnings(
            "ignore",
            message="The '.*_dataloader' does not have many workers.*",
            category=UserWarning,
        )
        if LightningDeprecationWarning is not None:
            warnings.filterwarnings("ignore", category=LightningDeprecationWarning)
