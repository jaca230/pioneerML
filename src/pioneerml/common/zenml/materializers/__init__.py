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
from pioneerml.common.zenml.materializers.group_classifier_event_dataset_materializer import (
    GroupClassifierEventDatasetMaterializer,
)
from pioneerml.common.zenml.materializers.group_splitter_dataset_materializer import (
    GroupSplitterDatasetMaterializer,
)
from pioneerml.common.zenml.materializers.group_splitter_event_dataset_materializer import (
    GroupSplitterEventDatasetMaterializer,
)
from pioneerml.common.zenml.materializers.endpoint_regressor_dataset_materializer import (
    EndpointRegressorDatasetMaterializer,
)
from pioneerml.common.zenml.materializers.endpoint_regressor_event_dataset_materializer import (
    EndpointRegressorEventDatasetMaterializer,
)
from pioneerml.common.zenml.materializers.event_splitter_event_dataset_materializer import (
    EventSplitterEventDatasetMaterializer,
)
from pioneerml.common.zenml.materializers.torch_tensor_materializer import (
    TorchTensorMaterializer,
)

source_utils.set_custom_source_root(Path(__file__).resolve().parents[4])

__all__ = [
    "GroupClassifierDatasetMaterializer",
    "GroupClassifierEventDatasetMaterializer",
    "GroupSplitterDatasetMaterializer",
    "GroupSplitterEventDatasetMaterializer",
    "EndpointRegressorDatasetMaterializer",
    "EndpointRegressorEventDatasetMaterializer",
    "EventSplitterEventDatasetMaterializer",
    "PyGDataListMaterializer",
    "TorchTensorMaterializer",
]
