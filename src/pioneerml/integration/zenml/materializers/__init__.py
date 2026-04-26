"""
Lightweight ZenML materializers to silence pickle warnings in tutorials.

These materializers keep artifacts tiny and easy to reload for the
synthetic tutorial pipelines.
"""

from __future__ import annotations

from pathlib import Path

from zenml.utils import source_utils

from pioneerml.integration.zenml.materializers.pyg_data_list_materializer import (
    PyGDataListMaterializer,
)
from pioneerml.integration.zenml.materializers.torch_tensor_materializer import (
    TorchTensorMaterializer,
)
from pioneerml.integration.zenml.materializers.batch_bundle_materializer import (
    BatchBundleMaterializer,
)
from pioneerml.integration.zenml.materializers.graph_lightning_module_materializer import (
    GraphLightningModuleMaterializer,
)
from pioneerml.integration.zenml.materializers.model_handle_materializer import (
    ModelHandleMaterializer,
)

source_utils.set_custom_source_root(Path(__file__).resolve().parents[4])

__all__ = [
    "BatchBundleMaterializer",
    "GraphLightningModuleMaterializer",
    "ModelHandleMaterializer",
    "PyGDataListMaterializer",
    "TorchTensorMaterializer",
]
