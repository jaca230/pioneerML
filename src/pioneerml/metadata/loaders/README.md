# Model Loaders

This directory contains model loaders for reconstructing models from checkpoints and metadata. The loader system uses a registry pattern to provide polymorphic model loading without hardcoded conditionals.

## Architecture

- **`base.py`**: Abstract base class `ModelLoader` that all loaders inherit from
- **`registry.py`**: Registry system that maps model types to loader classes
- **Individual loaders**: One file per model type (e.g., `group_classifier.py`, `pion_stop.py`)

## How It Works

1. Each model type has a loader class that inherits from `ModelLoader`
2. Loaders are automatically registered using the `@register_model_loader` decorator
3. The `load_model_from_checkpoint()` function uses the registry to find the appropriate loader
4. No hardcoded if/elif chains - everything is polymorphic

## Adding a New Model Type

To add support for a new model type:

1. **Create a new loader file** (e.g., `my_model.py`):
   ```python
   from pioneerml.metadata.manager import TrainingMetadata
   from pioneerml.models import MyModel
   from .base import ModelLoader
   from .registry import register_model_loader
   
   @register_model_loader("MyModel")
   class MyModelLoader(ModelLoader):
       @property
       def model_type(self) -> str:
           return "MyModel"
       
       def load_model(self, metadata: TrainingMetadata, *, device="cpu") -> torch.nn.Module:
           defaults = {"hidden": 128, "layers": 3, "dropout": 0.1}
           params = self.extract_architecture_params(metadata, defaults)
           
           return MyModel(
               hidden=int(params["hidden"]),
               layers=int(params["layers"]),
               dropout=float(params["dropout"]),
           )
   ```

2. **Import the module** in `loaders/__init__.py`:
   ```python
   from . import my_model  # noqa: E402, F401
   ```

3. **That's it!** The loader will be automatically registered and available.

## Usage

```python
from pioneerml.metadata import load_model_from_checkpoint

# Load any model type - no hardcoded conditionals needed!
model, metadata = load_model_from_checkpoint(
    model_type="MyModel",  # Works for any registered model type
    device="cuda",
)
```

## Benefits

- **Scalable**: Easy to add new model types without modifying existing code
- **Polymorphic**: No hardcoded if/elif chains
- **Organized**: Each model type has its own file
- **Testable**: Each loader can be tested independently
- **Type-safe**: Abstract base class ensures consistent interface


