"""
Utilities for ZenML notebook initialization.

These utilities help notebooks use the top-level .zenml and .zen store
configuration without needing to manually configure stores.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from zenml.client import Client
from zenml.config.global_config import GlobalConfiguration


def detect_available_accelerator():
    """
    Detect the best available accelerator for training.

    Returns:
        tuple: (accelerator, devices) where:
            - accelerator: "gpu", "mps", or "cpu"
            - devices: number of devices to use (typically 1)
    """
    import torch

    if torch.cuda.is_available():
        return "gpu", 1
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        # Apple Silicon GPU support
        return "mps", 1
    else:
        return "cpu", 1


def find_project_root(start_path: Path | None = None) -> Path:
    """
    Find the project root by searching upward for .zen or .zenml directory.
    
    This utility ensures we always use the root .zen/.zenml configuration
    regardless of where the notebook is executed from.
    
    Args:
        start_path: Starting path for search. If None, uses current working directory.
    
    Returns:
        Path to project root where .zen or .zenml exists.
    """
    if start_path is None:
        start_path = Path.cwd()
    else:
        start_path = Path(start_path)
    
    # Search upward for .zen or .zenml directory
    for path in [start_path] + list(start_path.parents):
        if (path / ".zen").exists() or (path / ".zenml").exists():
            return path
    
    # If not found, return the starting path
    return start_path


def setup_zenml_for_notebook(
    root_path: Path | str | None = None,
    use_in_memory: bool = True,
) -> Client:
    """
    Set up ZenML for use in notebooks using the project's global configuration.
    
    This function:
    1. Finds the project root (where .zen or .zenml exists) automatically
    2. Uses the existing global ZenML configuration
    3. Optionally switches to in-memory storage if requested
    
    Args:
        root_path: Path to project root. If None, automatically finds project root
            by searching upward for .zen or .zenml directory.
        use_in_memory: If True, use in-memory SQLite instead of file-based storage.
    
    Returns:
        Initialized ZenML Client
    """
    # Disable analytics for quickstart
    os.environ["ZENML_DISABLE_ANALYTICS"] = "true"
    
    # Find project root automatically if not provided
    if root_path is None:
        root_path = find_project_root()
    else:
        root_path = Path(root_path)
    
    # Get global configuration
    gc = GlobalConfiguration()
    
    # Optionally switch to in-memory storage
    if use_in_memory:
        try:
            store_config = gc.store_configuration
            # If there's a URL and it's not already in-memory, switch to in-memory
            if hasattr(store_config, 'url') and store_config.url:
                if ':memory:' not in store_config.url:
                    from zenml.config.store_config import StoreConfiguration
                    in_memory_store_config = StoreConfiguration(
                        type="sql",
                        url="sqlite:///:memory:"
                    )
                    gc.set_store(in_memory_store_config)
        except Exception:
            # If there's an error, try to create a fresh in-memory store
            try:
                from zenml.config.store_config import StoreConfiguration
                in_memory_store_config = StoreConfiguration(
                    type="sql",
                    url="sqlite:///:memory:"
                )
                gc.set_store(in_memory_store_config)
            except Exception:
                pass  # Continue with existing config
    
    # Initialize client - it will use the global configuration
    # The global config points to the .zen/.zenml in the project root
    try:
        client = Client(root=root_path)
        client.initialize()
    except Exception:
        # Already initialized or using existing config
        client = Client(root=root_path)
    
    return client


def load_step_output(
    run: Any,
    step_name: str,
    output_name: str = "output",
    index: int = 0,
) -> Any | None:
    """
    Load a step output artifact from a ZenML pipeline run.

    This helper keeps the notebook code tidy by centralizing the artifact
    loading logic and handling cases where the step/output is missing.

    Args:
        run: ZenML pipeline run object returned by executing a pipeline.
        step_name: Name of the step to load output from.
        output_name: Output key to load (defaults to \"output\").
            For tuple outputs, ZenML typically names them \"output_0\", \"output_1\", etc.
        index: Index within the output list (defaults to first).

    Returns:
        Loaded artifact object if available, otherwise ``None``.
    """
    try:
        step = getattr(run, "steps", {}).get(step_name)
        if step is None:
            return None

        # Try to get the output - ZenML may use different naming conventions
        artifacts = step.outputs.get(output_name)
        if not artifacts or len(artifacts) <= index:
            # Try alternative naming for tuple outputs
            if output_name == "output":
                # For tuple returns, try output_0, output_1, etc.
                alt_name = f"output_{index}"
                artifacts = step.outputs.get(alt_name)
                if not artifacts or len(artifacts) == 0:
                    return None
                artifact = artifacts[0] if artifacts else None
            else:
                return None
        else:
            artifact = artifacts[index]
        
        if artifact is None:
            return None
            
        if hasattr(artifact, "load"):
            return artifact.load()
    except Exception:
        return None

    return None
