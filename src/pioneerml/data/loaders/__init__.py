"""
Data loaders.

Legacy NPY loaders are deprecated; prefer parquet processors.
"""

from .base import BaseLoader
from .base import BaseLoader

# Legacy shim to avoid hard import breaks
def load_hits_and_info(*args, **kwargs):
    import warnings
    from pioneerml.deprecated.loaders.hits_info import load_hits_and_info as _deprecated_loader

    warnings.warn("load_hits_and_info is deprecated; switch to parquet processors.", DeprecationWarning, stacklevel=2)
    return _deprecated_loader(*args, **kwargs)


__all__ = [
    "BaseLoader",
    "load_hits_and_info",
]
