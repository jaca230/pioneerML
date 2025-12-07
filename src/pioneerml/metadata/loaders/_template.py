"""
Template for creating a new model loader.

To add a new model type:
1. Copy this file and rename it (e.g., `my_model.py`)
2. Replace `MyModel` with your actual model class name
3. Replace `"MyModel"` with your model type string
4. Implement the `load_model()` method to extract parameters and instantiate your model
5. Import this module in `loaders/__init__.py` to trigger registration

Example:
    @register_model_loader("MyModel")
    class MyModelLoader(ModelLoader):
        @property
        def model_type(self) -> str:
            return "MyModel"
        
        def load_model(
            self,
            metadata: TrainingMetadata,
            *,
            device: str | torch.device = "cpu",
        ) -> torch.nn.Module:
            # Extract parameters from metadata
            defaults = {
                "param1": default_value1,
                "param2": default_value2,
                # ... add all parameters your model needs
            }
            
            params = self.extract_architecture_params(metadata, defaults)
            
            # Apply any model-specific transformations
            # (e.g., ensuring hidden is divisible by heads)
            
            # Instantiate and return the model
            from pioneerml.models import MyModel
            return MyModel(
                param1=int(params["param1"]),
                param2=float(params["param2"]),
                # ... pass all parameters
            )
"""

from __future__ import annotations

import torch

from pioneerml.metadata.types import TrainingMetadata
# from pioneerml.models import MyModel  # Import your model class here
from .base import ModelLoader
from .registry import register_model_loader


@register_model_loader("MyModel")  # Replace with your model type string
class MyModelLoader(ModelLoader):
    """Loader for MyModel models."""
    
    @property
    def model_type(self) -> str:
        return "MyModel"  # Replace with your model type string
    
    def load_model(
        self,
        metadata: TrainingMetadata,
        *,
        device: str | torch.device = "cpu",
    ) -> torch.nn.Module:
        """
        Load a MyModel from metadata.
        
        This method should:
        1. Extract architecture parameters from metadata
        2. Apply any model-specific transformations
        3. Instantiate the model with those parameters
        """
        # Define default values for all parameters
        defaults = {
            "hidden": 128,
            "layers": 3,
            "dropout": 0.1,
            # Add all parameters your model needs
        }
        
        # Extract parameters from metadata (tries architecture, then best_params, then defaults)
        params = self.extract_architecture_params(metadata, defaults)
        
        # Apply any model-specific logic here
        # For example, ensuring hidden is divisible by heads:
        # hidden = int(params["hidden"])
        # heads = int(params["heads"])
        # hidden = (hidden // heads) * heads
        
        # Import your model class
        # from pioneerml.models import MyModel
        
        # Instantiate and return the model
        # return MyModel(
        #     hidden=int(params["hidden"]),
        #     layers=int(params["layers"]),
        #     dropout=float(params["dropout"]),
        #     # ... pass all parameters
        # )
        
        raise NotImplementedError("Implement this method for your model type")

