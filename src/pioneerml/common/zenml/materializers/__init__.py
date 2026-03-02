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
from pioneerml.common.zenml.materializers.torch_tensor_materializer import (
    TorchTensorMaterializer,
)
from pioneerml.common.zenml.materializers.inference_batch_bundle_materializer import (
    InferenceBatchBundleMaterializer,
)
from pioneerml.common.zenml.materializers.training_batch_bundle_materializer import (
    TrainingBatchBundleMaterializer,
)
from pioneerml.common.zenml.materializers.graph_lightning_module_materializer import (
    GraphLightningModuleMaterializer,
)

source_utils.set_custom_source_root(Path(__file__).resolve().parents[4])

__all__ = [
    "TrainingBatchBundleMaterializer",
    "InferenceBatchBundleMaterializer",
    "GraphLightningModuleMaterializer",
    "PyGDataListMaterializer",
    "TorchTensorMaterializer",
]
