"""
Lightweight ZenML materializers to silence pickle warnings in tutorials.

These materializers keep artifacts tiny and easy to reload for the
synthetic tutorial pipelines.
"""

from __future__ import annotations

from pathlib import Path

from zenml.utils import source_utils

from pioneerml.common.zenml.materializers.endpoint_regressor_dataset_materializer import (
    EndpointRegressorDatasetMaterializer,
)
from pioneerml.common.zenml.materializers.event_splitter_dataset_materializer import (
    EventSplitterDatasetMaterializer,
)
from pioneerml.common.zenml.materializers.group_classifier_dataset_materializer import (
    GroupClassifierDatasetMaterializer,
)
from pioneerml.common.zenml.materializers.group_splitter_dataset_materializer import (
    GroupSplitterDatasetMaterializer,
)
from pioneerml.common.zenml.materializers.pion_stop_dataset_materializer import (
    PionStopDatasetMaterializer,
)
from pioneerml.common.zenml.materializers.positron_angle_dataset_materializer import (
    PositronAngleDatasetMaterializer,
)
from pioneerml.common.zenml.materializers.pyg_data_list_materializer import (
    PyGDataListMaterializer,
)
from pioneerml.common.zenml.materializers.torch_tensor_materializer import (
    TorchTensorMaterializer,
)

try:
    from pioneerml.common.zenml.materializers.group_classifier_event_dataset_materializer import (
        GroupClassifierEventDatasetMaterializer,
    )
except Exception:  # pragma: no cover - optional legacy materializer
    GroupClassifierEventDatasetMaterializer = None

try:
    from pioneerml.common.zenml.materializers.group_splitter_event_dataset_materializer import (
        GroupSplitterEventDatasetMaterializer,
    )
except Exception:  # pragma: no cover - optional legacy materializer
    GroupSplitterEventDatasetMaterializer = None

try:
    from pioneerml.common.zenml.materializers.endpoint_regressor_event_dataset_materializer import (
        EndpointRegressorEventDatasetMaterializer,
    )
except Exception:  # pragma: no cover - optional legacy materializer
    EndpointRegressorEventDatasetMaterializer = None

try:
    from pioneerml.common.zenml.materializers.event_splitter_event_dataset_materializer import (
        EventSplitterEventDatasetMaterializer,
    )
except Exception:  # pragma: no cover - optional legacy materializer
    EventSplitterEventDatasetMaterializer = None

source_utils.set_custom_source_root(Path(__file__).resolve().parents[4])

__all__ = [
    "GroupClassifierDatasetMaterializer",
    "GroupSplitterDatasetMaterializer",
    "EndpointRegressorDatasetMaterializer",
    "EventSplitterDatasetMaterializer",
    "PionStopDatasetMaterializer",
    "PositronAngleDatasetMaterializer",
    "PyGDataListMaterializer",
    "TorchTensorMaterializer",
]
if GroupClassifierEventDatasetMaterializer is not None:
    __all__.append("GroupClassifierEventDatasetMaterializer")
if GroupSplitterEventDatasetMaterializer is not None:
    __all__.append("GroupSplitterEventDatasetMaterializer")
if EndpointRegressorEventDatasetMaterializer is not None:
    __all__.append("EndpointRegressorEventDatasetMaterializer")
if EventSplitterEventDatasetMaterializer is not None:
    __all__.append("EventSplitterEventDatasetMaterializer")
