"""
Model loaders for reconstructing models from checkpoints and metadata.

Each model type has its own loader class that knows how to:
1. Extract architecture parameters from metadata
2. Instantiate the model with those parameters
3. Handle any model-specific logic

To add a new model type:
1. Create a new loader class in a new file (e.g., `my_model.py`)
2. Inherit from `ModelLoader` and implement `load_model()`
3. Register it using `register_loader()` or the `@register_model_loader` decorator
4. Import the module in this __init__.py to trigger registration
"""

from .base import ModelLoader
from .registry import (
    MODEL_LOADER_REGISTRY,
    get_loader,
    load_model_from_checkpoint,
    register_loader,
    register_model_loader,
)

# Import all loaders to trigger registration
# These imports must come after the registry imports
from . import group_classifier  # noqa: E402, F401
from . import group_splitter  # noqa: E402, F401
from . import pion_stop  # noqa: E402, F401
from . import positron_angle  # noqa: E402, F401

__all__ = [
    "ModelLoader",
    "MODEL_LOADER_REGISTRY",
    "get_loader",
    "load_model_from_checkpoint",
    "register_loader",
    "register_model_loader",
]

