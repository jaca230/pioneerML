"""
Pipeline Context for shared state between stages.

The Context is a key-value store that stages use to communicate.
It supports various data types and provides utilities for serialization.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Set
from pathlib import Path
import pickle
import json


class Context:
    """
    Shared state container for pipeline stages.

    Context acts as shared memory that all stages can read from and write to.
    It's essentially a type-aware dictionary with utilities for common operations.

    Features:
    - Dictionary-like interface (get/set/contains)
    - Type tracking for debugging
    - Serialization support
    - History tracking (optional)
    - Nested key access with dot notation

    Example:
        >>> ctx = Context()
        >>> ctx['dataset'] = load_data()
        >>> ctx['model'] = GroupClassifier()
        >>> ctx.set('metrics.accuracy', 0.95)
        >>> print(ctx.get('metrics.accuracy'))
        0.95
    """

    def __init__(self, initial_data: Optional[Dict[str, Any]] = None):
        """
        Initialize context.

        Args:
            initial_data: Optional initial key-value pairs.
        """
        self._data: Dict[str, Any] = initial_data or {}
        self._history: List[tuple] = []  # (key, operation, value_type)
        self._track_history = False

    def __getitem__(self, key: str) -> Any:
        """Get value by key."""
        return self._data[key]

    def __setitem__(self, key: str, value: Any) -> None:
        """Set value by key."""
        if self._track_history:
            self._history.append((key, "set", type(value).__name__))
        self._data[key] = value

    def __contains__(self, key: str) -> bool:
        """Check if key exists."""
        return key in self._data

    def __delitem__(self, key: str) -> None:
        """Delete key."""
        if self._track_history:
            self._history.append((key, "del", None))
        del self._data[key]

    def __len__(self) -> int:
        """Number of keys in context."""
        return len(self._data)

    def __repr__(self) -> str:
        """String representation."""
        keys = list(self._data.keys())
        return f"Context(keys={keys})"

    def get(self, key: str, default: Any = None) -> Any:
        """
        Get value with optional default.

        Supports dot notation for nested access:
            ctx.get('model.hidden_size')

        Args:
            key: Key to retrieve (supports dots for nesting).
            default: Default value if key not found.

        Returns:
            Value associated with key, or default.
        """
        if "." not in key:
            return self._data.get(key, default)

        # Handle nested keys
        parts = key.split(".")
        obj = self._data
        for part in parts[:-1]:
            if not isinstance(obj, dict) or part not in obj:
                return default
            obj = obj[part]

        return obj.get(parts[-1], default) if isinstance(obj, dict) else default

    def set(self, key: str, value: Any) -> None:
        """
        Set value (supports dot notation for nesting).

        Args:
            key: Key to set (supports dots for nesting).
            value: Value to store.
        """
        if "." not in key:
            self[key] = value
            return

        # Handle nested keys - create nested dicts as needed
        parts = key.split(".")
        obj = self._data
        for part in parts[:-1]:
            if part not in obj:
                obj[part] = {}
            obj = obj[part]

        obj[parts[-1]] = value
        if self._track_history:
            self._history.append((key, "set", type(value).__name__))

    def update(self, other: Dict[str, Any]) -> None:
        """
        Update context with multiple key-value pairs.

        Args:
            other: Dictionary of updates.
        """
        for key, value in other.items():
            self[key] = value

    def keys(self) -> List[str]:
        """Get all keys in context."""
        return list(self._data.keys())

    def values(self) -> List[Any]:
        """Get all values in context."""
        return list(self._data.values())

    def items(self) -> List[tuple]:
        """Get all key-value pairs."""
        return list(self._data.items())

    def pop(self, key: str, default: Any = None) -> Any:
        """
        Remove and return value.

        Args:
            key: Key to remove.
            default: Default if key not found.

        Returns:
            Value that was removed, or default.
        """
        if self._track_history:
            self._history.append((key, "pop", None))
        return self._data.pop(key, default)

    def clear(self) -> None:
        """Clear all data from context."""
        if self._track_history:
            self._history.append(("*", "clear", None))
        self._data.clear()

    def enable_history(self) -> None:
        """Enable history tracking of operations."""
        self._track_history = True

    def disable_history(self) -> None:
        """Disable history tracking."""
        self._track_history = False

    def get_history(self) -> List[tuple]:
        """
        Get history of operations.

        Returns:
            List of (key, operation, value_type) tuples.
        """
        return self._history.copy()

    def save(self, path: str | Path) -> None:
        """
        Save context to disk.

        Note: Only picklable objects can be saved.

        Args:
            path: Path to save file.
        """
        path = Path(path)
        with open(path, "wb") as f:
            pickle.dump(self._data, f)

    @classmethod
    def load(cls, path: str | Path) -> Context:
        """
        Load context from disk.

        Args:
            path: Path to load from.

        Returns:
            Loaded Context instance.
        """
        path = Path(path)
        with open(path, "rb") as f:
            data = pickle.load(f)
        return cls(initial_data=data)

    def summary(self) -> Dict[str, Any]:
        """
        Get summary of context contents.

        Returns:
            Dictionary with keys, types, and sizes.
        """
        summary = {}
        for key, value in self._data.items():
            summary[key] = {
                "type": type(value).__name__,
                "size": self._estimate_size(value),
            }
        return summary

    @staticmethod
    def _estimate_size(obj: Any) -> str:
        """
        Estimate size of object.

        Args:
            obj: Object to measure.

        Returns:
            Human-readable size string.
        """
        import sys

        size = sys.getsizeof(obj)

        # Add size of contents for containers
        if hasattr(obj, "__len__"):
            try:
                size += sum(sys.getsizeof(item) for item in obj)
            except (TypeError, AttributeError):
                pass

        # Convert to human readable
        for unit in ["B", "KB", "MB", "GB"]:
            if size < 1024.0:
                return f"{size:.2f} {unit}"
            size /= 1024.0

        return f"{size:.2f} TB"

    def copy(self) -> Context:
        """
        Create a shallow copy of the context.

        Returns:
            New Context with copied data.
        """
        return Context(initial_data=self._data.copy())

    def deep_copy(self) -> Context:
        """
        Create a deep copy of the context.

        Returns:
            New Context with deeply copied data.
        """
        import copy

        return Context(initial_data=copy.deepcopy(self._data))
