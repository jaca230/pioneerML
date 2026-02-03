"""
Lightweight ZenML materializers to silence pickle warnings in tutorials.

These materializers keep artifacts tiny and easy to reload for the
synthetic tutorial pipelines.
"""

from __future__ import annotations

from pathlib import Path

from zenml.utils import source_utils

from pioneerml.common.zenml.materializers.pyg_data_list_materializer import (
    PyGDataListMaterializer,
)
from pioneerml.common.zenml.materializers.group_classifier_dataset_materializer import (
    GroupClassifierDatasetMaterializer,
)
from pioneerml.common.zenml.materializers.torch_tensor_materializer import (
    TorchTensorMaterializer,
)

source_utils.set_custom_source_root(Path(__file__).resolve().parents[4])

__all__ = [
    "GroupClassifierDatasetMaterializer",
    "PyGDataListMaterializer",
    "TorchTensorMaterializer",
]
