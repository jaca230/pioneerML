from __future__ import annotations

import logging
import warnings

from .base_log_filter import BaseLogFilter
from .factory.registry import REGISTRY as LOG_FILTER_REGISTRY

try:
    from lightning_fabric.utilities.rank_zero import LightningDeprecationWarning
except Exception:  # pragma: no cover
    LightningDeprecationWarning = None


@LOG_FILTER_REGISTRY.register("training")
class TrainingLogFilter(BaseLogFilter):
    def apply(self) -> None:
        suppress_litlogger_tip = bool(self.config.get("suppress_litlogger_tip", True))
        suppress_treespec_deprecation = bool(self.config.get("suppress_treespec_deprecation", True))
        suppress_dataloader_workers_warning = bool(self.config.get("suppress_dataloader_workers_warning", True))
        suppress_lightning_deprecation = bool(self.config.get("suppress_lightning_deprecation", True))

        if suppress_litlogger_tip:
            self._suppress_litlogger_tip()

        if suppress_treespec_deprecation:
            self._suppress_treespec_deprecation()

        if suppress_dataloader_workers_warning:
            self._suppress_dataloader_workers_warning()

        if suppress_lightning_deprecation and LightningDeprecationWarning is not None:
            self._suppress_lightning_deprecation()

    @staticmethod
    def _suppress_litlogger_tip() -> None:
        class _DropLitloggerTip(logging.Filter):
            def filter(self, record: logging.LogRecord) -> bool:
                msg = record.getMessage()
                return (
                    "For seamless cloud logging and experiment tracking, try installing [litlogger]"
                    not in msg
                )

        tip_filter = _DropLitloggerTip()
        for logger_name in (
            "lightning.pytorch.utilities.rank_zero",
            "pytorch_lightning.utilities.rank_zero",
        ):
            logging.getLogger(logger_name).addFilter(tip_filter)

    @staticmethod
    def _suppress_treespec_deprecation() -> None:
        warnings.filterwarnings("ignore", message="isinstance\\(treespec, LeafSpec\\) is deprecated.*")
        warnings.filterwarnings(
            "ignore",
            message="`isinstance\\(treespec, LeafSpec\\)` is deprecated.*",
            category=FutureWarning,
        )

    @staticmethod
    def _suppress_dataloader_workers_warning() -> None:
        warnings.filterwarnings(
            "ignore",
            message="The '.*_dataloader' does not have many workers.*",
            category=UserWarning,
        )

    @staticmethod
    def _suppress_lightning_deprecation() -> None:
        warnings.filterwarnings("ignore", category=LightningDeprecationWarning)
