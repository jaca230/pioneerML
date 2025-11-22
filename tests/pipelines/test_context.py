"""
Tests for Pipeline Context.
"""

import pytest
from pioneerml.pipelines import Context


class TestContext:
    """Tests for Context class."""

    def test_init_empty(self):
        """Test creating empty context."""
        ctx = Context()
        assert len(ctx) == 0

    def test_init_with_data(self):
        """Test creating context with initial data."""
        data = {"a": 1, "b": 2}
        ctx = Context(initial_data=data)
        assert len(ctx) == 2
        assert ctx["a"] == 1
        assert ctx["b"] == 2

    def test_getitem_setitem(self):
        """Test getting and setting items."""
        ctx = Context()
        ctx["key"] = "value"
        assert ctx["key"] == "value"

    def test_contains(self):
        """Test checking if key exists."""
        ctx = Context()
        ctx["present"] = 1
        assert "present" in ctx
        assert "absent" not in ctx

    def test_get_with_default(self):
        """Test get with default value."""
        ctx = Context()
        assert ctx.get("missing", "default") == "default"
        ctx["present"] = "value"
        assert ctx.get("present", "default") == "value"

    def test_nested_get_set(self):
        """Test nested key access with dot notation."""
        ctx = Context()

        # Set nested value
        ctx.set("model.hidden_size", 128)
        assert ctx.get("model.hidden_size") == 128

        # Access via dict
        assert ctx["model"]["hidden_size"] == 128

    def test_update(self):
        """Test updating multiple keys."""
        ctx = Context()
        ctx.update({"a": 1, "b": 2, "c": 3})
        assert len(ctx) == 3
        assert ctx["a"] == 1

    def test_keys_values_items(self):
        """Test getting keys, values, and items."""
        ctx = Context({"a": 1, "b": 2})
        assert set(ctx.keys()) == {"a", "b"}
        assert set(ctx.values()) == {1, 2}
        assert set(ctx.items()) == {("a", 1), ("b", 2)}

    def test_pop(self):
        """Test popping values."""
        ctx = Context({"a": 1, "b": 2})
        val = ctx.pop("a")
        assert val == 1
        assert "a" not in ctx
        assert ctx.pop("missing", "default") == "default"

    def test_clear(self):
        """Test clearing context."""
        ctx = Context({"a": 1, "b": 2})
        ctx.clear()
        assert len(ctx) == 0

    def test_history_tracking(self):
        """Test history tracking of operations."""
        ctx = Context()
        ctx.enable_history()

        ctx["a"] = 1
        ctx["b"] = "test"
        ctx.pop("a")

        history = ctx.get_history()
        assert len(history) >= 2
        assert history[0][0] == "a"  # key
        assert history[0][1] == "set"  # operation

    def test_summary(self):
        """Test getting summary of context."""
        ctx = Context()
        ctx["numbers"] = [1, 2, 3]
        ctx["text"] = "hello"

        summary = ctx.summary()
        assert "numbers" in summary
        assert "text" in summary
        assert summary["numbers"]["type"] == "list"
        assert summary["text"]["type"] == "str"

    def test_copy(self):
        """Test shallow copy."""
        ctx = Context({"a": 1, "b": [1, 2, 3]})
        ctx2 = ctx.copy()

        assert ctx2["a"] == 1
        assert ctx2["b"] == [1, 2, 3]

        # Modify original
        ctx["a"] = 999
        assert ctx2["a"] == 1  # Should not change

    def test_repr(self):
        """Test string representation."""
        ctx = Context({"a": 1, "b": 2})
        repr_str = repr(ctx)
        assert "Context" in repr_str
        assert "keys=" in repr_str
