"""
Utilities for ZenML notebook initialization.

These utilities help notebooks use the top-level .zenml and .zen store
configuration without needing to manually configure stores.
"""

from __future__ import annotations

import builtins
import os
from pathlib import Path
from typing import Any

import torch
from zenml.client import Client
from zenml.config.global_config import GlobalConfiguration

try:
    import torch_xla  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    torch_xla = None

try:
    from zenml.config.store_config import StoreConfiguration  # type: ignore
except Exception:  # pragma: no cover - optional optional import
    StoreConfiguration = None


def detect_available_accelerator():
    """
    Detect the best available accelerator for training.

    Returns:
        tuple: (accelerator, devices) where:
            - accelerator: "tpu", "gpu", "mps", or "cpu"
            - devices: number of devices to use (typically 1)
    """
    # Check for TPU first (highest priority)
    if torch_xla is not None:
        try:
            if torch_xla._XLAC._xla_get_default_device() != "CPU":  # type: ignore[attr-defined]
                return "tpu", 1
        except (AttributeError, RuntimeError):
            pass

    if torch.cuda.is_available():
        return "gpu", 1
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        # Apple Silicon GPU support
        return "mps", 1
    else:
        return "cpu", 1


def find_project_root(start: Path | None = None) -> Path:
    """
    Locate the project root by walking upward until a known sentinel file/dir
    is found.

    Sentinel order (can add more if needed):
      - pyproject.toml : modern Python project root marker (preferred)

    Args:
        start:
            Directory to begin searching from. Defaults to the current working
            directory.

    Returns:
        Path:
            The detected project root. If no sentinel is found, returns the
            starting directory.
    """
    # Use cwd if no starting path is provided
    path = Path(start or Path.cwd()).resolve()

    # Define sentinels locally (no need for global constants)
    sentinels = [
        "pyproject.toml",
    ]

    # Walk upward: starting directory + all parent directories
    for parent in [path, *path.parents]:
        if any((parent / s).exists() for s in sentinels):
            return parent

    # Fallback if nothing is found
    return path


def setup_zenml_for_notebook(
    root_path: Path | str | None = None,
    use_in_memory: bool = True,
    restore_original_print: bool = True,
) -> Client:
    """
    Set up ZenML for use in notebooks using the proper global Client behavior.

    This version follows ZenML's documented API:
    - Client().activate_root(path) is the ONLY supported way to set the repo root.
    """

    os.environ["ZENML_DISABLE_ANALYTICS"] = "true"

    # Determine repo root
    if root_path is None:
        root_path = find_project_root()
    root_path = Path(root_path).resolve()

    print(
        f"Using ZenML repository root: {root_path}\n"
        f"Ensure this is the top-level of your repo (.zen must live here)."
    )

    # Switch to in-memory store if requested
    if use_in_memory:
        gc = GlobalConfiguration()
        if StoreConfiguration is not None:
            try:
                gc.set_store(StoreConfiguration(type="sql", url="sqlite:///:memory:"))
            except Exception:
                pass  # fallback silently

    # activate_root updates the *global* Client singleton
    Client().activate_root(root_path)

    # ZenML monkey-patches builtins.print in some environments and currently
    # duplicates positional args > 1. Restore original print for notebook UX.
    if restore_original_print:
        original_print = getattr(builtins, "_zenml_original_print", None)
        if original_print is not None and builtins.print is not original_print:
            builtins.print = original_print

    # Return the global client
    return Client()



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
