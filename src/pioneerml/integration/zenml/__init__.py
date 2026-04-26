"""
ZenML integration for PIONEER ML.

This module provides utilities for working with ZenML pipelines and
configuring ZenML for notebook use.
"""

from pioneerml.integration.zenml import utils
from pioneerml.integration.zenml import materializers as _materializers
from pioneerml.integration.zenml.utils import (
    detect_available_accelerator,
    load_step_output,
    setup_repo_pythonpath,
)

# Best-effort bootstrap so plugin imports work even if user imports plugin
# modules before calling `setup_zenml_for_notebook(...)`.
try:
    setup_repo_pythonpath()
except Exception:
    pass

__all__ = [
    "utils",
    "detect_available_accelerator",
    "load_step_output",
    "setup_repo_pythonpath",
]
