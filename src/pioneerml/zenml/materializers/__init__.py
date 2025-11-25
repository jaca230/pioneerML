"""
Lightweight ZenML materializers to silence pickle warnings in tutorials.

These materializers keep artifacts tiny and easy to reload for the
synthetic tutorial pipelines.
"""

from __future__ import annotations

from pathlib import Path

from zenml.utils import source_utils

from pioneerml.zenml.materializers.graph_data_module_materializer import (
    GraphDataModuleMaterializer,
)
from pioneerml.zenml.materializers.pyg_data_list_materializer import (
    PyGDataListMaterializer,
)
from pioneerml.zenml.materializers.torch_tensor_materializer import (
    TorchTensorMaterializer,
)

source_utils.set_custom_source_root(Path(__file__).resolve().parents[3])

__all__ = [
    "GraphDataModuleMaterializer",
    "PyGDataListMaterializer",
    "TorchTensorMaterializer",
]
