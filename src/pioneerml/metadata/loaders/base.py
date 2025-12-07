"""
Abstract base class for model loaders.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict

import torch

from pioneerml.metadata.types import TrainingMetadata


class ModelLoader(ABC):
    """
    Abstract base class for loading models from checkpoints.
    
    Each model type should have a loader that inherits from this class
    and implements the `load_model()` method.
    """
    
    @property
    @abstractmethod
    def model_type(self) -> str:
        """Return the model type string (e.g., 'GroupClassifier')."""
        pass
    
    @abstractmethod
    def load_model(
        self,
        metadata: TrainingMetadata,
        *,
        device: str | torch.device = "cpu",
    ) -> torch.nn.Module:
        """
        Load and instantiate a model from metadata.
        
        Args:
            metadata: TrainingMetadata containing model architecture info
            device: Device to load model onto
        
        Returns:
            Instantiated model (not yet loaded with state dict)
        """
        pass
    
    def extract_architecture_params(
        self,
        metadata: TrainingMetadata,
        defaults: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Extract architecture parameters from metadata with fallback to defaults.
        
        This is a helper method that loaders can use to extract parameters
        from metadata.model_architecture or metadata.best_hyperparameters.
        
        Args:
            metadata: TrainingMetadata object
            defaults: Dictionary of default parameter values
        
        Returns:
            Dictionary of extracted parameters
        """
        arch = metadata.model_architecture or {}
        best_params = metadata.best_hyperparameters or {}
        
        params = {}
        for key, default_value in defaults.items():
            # Try architecture first, then best_params, then default
            params[key] = arch.get(key) or best_params.get(key) or default_value
        
        return params

