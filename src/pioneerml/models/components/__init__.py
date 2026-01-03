"""
Shared model components (stereo encoder, event builder) re-exported for clarity.
"""

from pioneerml.models.components.event_builder import EventBuilder  # noqa: F401
from pioneerml.models.components.stereo import (  # noqa: F401
    ViewAwareEncoder,
    QuantileOutputHead,
    VIEW_X_VAL,
    VIEW_Y_VAL,
)

__all__ = [
    "EventBuilder",
    "ViewAwareEncoder",
    "QuantileOutputHead",
    "VIEW_X_VAL",
    "VIEW_Y_VAL",
]
