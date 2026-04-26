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
        suppress_tensor_core_tip = bool(self.config.get("suppress_tensor_core_tip", True))
        suppress_treespec_deprecation = bool(self.config.get("suppress_treespec_deprecation", True))
        suppress_dataloader_workers_warning = bool(self.config.get("suppress_dataloader_workers_warning", True))
        suppress_lightning_deprecation = bool(self.config.get("suppress_lightning_deprecation", True))
        suppress_model_checkpoint_monitor_warning = bool(
            self.config.get("suppress_model_checkpoint_monitor_warning", True)
        )
        suppress_transformer_mask_type_warning = bool(
            self.config.get("suppress_transformer_mask_type_warning", True)
        )
        suppress_nested_tensor_prototype_warning = bool(
            self.config.get("suppress_nested_tensor_prototype_warning", True)
        )
        suppress_torch_geometric_scatter_hint = bool(
            self.config.get("suppress_torch_geometric_scatter_hint", True)
        )

        if suppress_litlogger_tip:
            self._suppress_litlogger_tip()

        if suppress_tensor_core_tip:
            self._suppress_tensor_core_tip()

        if suppress_treespec_deprecation:
            self._suppress_treespec_deprecation()

        if suppress_dataloader_workers_warning:
            self._suppress_dataloader_workers_warning()

        if suppress_lightning_deprecation and LightningDeprecationWarning is not None:
            self._suppress_lightning_deprecation()

        if suppress_model_checkpoint_monitor_warning:
            self._suppress_model_checkpoint_monitor_warning()

        if suppress_transformer_mask_type_warning:
            self._suppress_transformer_mask_type_warning()

        if suppress_nested_tensor_prototype_warning:
            self._suppress_nested_tensor_prototype_warning()

        if suppress_torch_geometric_scatter_hint:
            self._suppress_torch_geometric_scatter_hint()

    @staticmethod
    def _suppress_litlogger_tip() -> None:
        class _DropLitloggerTip(logging.Filter):
            def filter(self, record: logging.LogRecord) -> bool:
                msg = record.getMessage()
                lower_msg = msg.lower()
                is_tip = ("litlogger" in lower_msg) and (
                    ("cloud logging" in lower_msg) or ("experiment tracking" in lower_msg)
                )
                return not is_tip

        tip_filter = _DropLitloggerTip()
        for logger_name in (
            "",
            "lightning_fabric.utilities.rank_zero",
            "lightning.fabric.utilities.rank_zero",
            "lightning.pytorch.utilities.rank_zero_info",
            "lightning.pytorch.utilities.rank_zero",
            "pytorch_lightning.utilities.rank_zero",
            "lightning",
        ):
            logging.getLogger(logger_name).addFilter(tip_filter)

    @staticmethod
    def _suppress_tensor_core_tip() -> None:
        warnings.filterwarnings(
            "ignore",
            message=r".*Tensor Cores.*torch\.set_float32_matmul_precision.*",
            category=UserWarning,
        )

        class _DropTensorCoreTip(logging.Filter):
            def filter(self, record: logging.LogRecord) -> bool:
                msg = record.getMessage()
                is_tip = ("Tensor Cores" in msg) and ("matmul_precision" in msg)
                return not is_tip

        tensor_core_filter = _DropTensorCoreTip()
        for logger_name in (
            "",
            "lightning_fabric.utilities.rank_zero",
            "lightning.fabric.utilities.rank_zero",
            "lightning.pytorch.utilities.rank_zero_info",
            "lightning.pytorch.utilities.rank_zero",
            "pytorch_lightning.utilities.rank_zero",
            "lightning",
        ):
            logging.getLogger(logger_name).addFilter(tensor_core_filter)

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

    @staticmethod
    def _suppress_model_checkpoint_monitor_warning() -> None:
        warnings.filterwarnings(
            "ignore",
            message=r".*could not find the monitored key.*",
        )

        class _DropCheckpointMonitorMissing(logging.Filter):
            def filter(self, record: logging.LogRecord) -> bool:
                msg = record.getMessage()
                is_model_ckpt_warning = "ModelCheckpoint(" in msg and "could not find the monitored key" in msg
                return not is_model_ckpt_warning

        monitor_filter = _DropCheckpointMonitorMissing()
        for logger_name in (
            "lightning.pytorch.callbacks.model_checkpoint",
            "pytorch_lightning.callbacks.model_checkpoint",
            "lightning.pytorch.utilities.rank_zero",
            "pytorch_lightning.utilities.rank_zero",
        ):
            logging.getLogger(logger_name).addFilter(monitor_filter)

        logging.getLogger("py.warnings").addFilter(monitor_filter)

    @staticmethod
    def _suppress_transformer_mask_type_warning() -> None:
        warnings.filterwarnings(
            "ignore",
            message=r"Support for mismatched key_padding_mask and attn_mask is deprecated.*",
            category=UserWarning,
        )
        warnings.filterwarnings(
            "ignore",
            message=r"Support for mismatched src_key_padding_mask and mask is deprecated.*",
            category=UserWarning,
        )

    @staticmethod
    def _suppress_nested_tensor_prototype_warning() -> None:
        warnings.filterwarnings(
            "ignore",
            message=r"The PyTorch API of nested tensors is in prototype stage.*",
            category=UserWarning,
        )

    @staticmethod
    def _suppress_torch_geometric_scatter_hint() -> None:
        warnings.filterwarnings(
            "ignore",
            message=r".*torch-scatter.*not found.*",
            category=UserWarning,
        )
