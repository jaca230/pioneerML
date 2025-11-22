"""
Tests for Pipeline Stage classes.
"""

import pytest
from pioneerml.pipelines import Stage, StageConfig, FunctionalStage, Context


class DummyStage(Stage):
    """Dummy stage for testing."""

    def execute(self, context):
        """Simple execution."""
        context["executed"] = True


class TestStageConfig:
    """Tests for StageConfig."""

    def test_basic_config(self):
        """Test creating basic config."""
        config = StageConfig(name="test_stage", inputs=["a"], outputs=["b"])

        assert config.name == "test_stage"
        assert config.inputs == ["a"]
        assert config.outputs == ["b"]
        assert config.enabled is True

    def test_with_params(self):
        """Test config with parameters."""
        config = StageConfig(
            name="test",
            inputs=["x"],
            outputs=["y"],
            params={"lr": 0.001, "batch_size": 32},
        )

        assert config.params["lr"] == 0.001
        assert config.params["batch_size"] == 32


class TestStage:
    """Tests for Stage base class."""

    def test_stage_initialization(self):
        """Test initializing a stage."""
        config = StageConfig(name="dummy", inputs=["a"], outputs=["b"])
        stage = DummyStage(config)

        assert stage.name == "dummy"
        assert stage.inputs == ["a"]
        assert stage.outputs == ["b"]
        assert stage.enabled is True

    def test_stage_execution(self):
        """Test executing a stage."""
        config = StageConfig(name="dummy", outputs=["executed"])
        stage = DummyStage(config)

        ctx = Context()
        stage.setup(ctx)
        stage.execute(ctx)
        stage.cleanup(ctx)

        assert ctx["executed"] is True
        assert stage._is_setup is True
        assert stage._is_cleaned_up is True

    def test_validate_inputs_success(self):
        """Test input validation when inputs exist."""
        config = StageConfig(name="dummy", inputs=["a", "b"])
        stage = DummyStage(config)

        ctx = Context({"a": 1, "b": 2})
        stage.validate_inputs(ctx)  # Should not raise

    def test_validate_inputs_failure(self):
        """Test input validation when inputs missing."""
        config = StageConfig(name="dummy", inputs=["a", "b"])
        stage = DummyStage(config)

        ctx = Context({"a": 1})  # Missing 'b'

        with pytest.raises(KeyError, match="missing required inputs"):
            stage.validate_inputs(ctx)

    def test_stage_repr(self):
        """Test string representation."""
        config = StageConfig(name="test", inputs=["x"], outputs=["y"])
        stage = DummyStage(config)

        repr_str = repr(stage)
        assert "DummyStage" in repr_str
        assert "test" in repr_str


class TestFunctionalStage:
    """Tests for FunctionalStage."""

    def test_functional_stage(self):
        """Test wrapping a function as a stage."""

        def my_func(ctx):
            ctx["result"] = ctx["input"] * 2

        config = StageConfig(name="double", inputs=["input"], outputs=["result"])
        stage = FunctionalStage(config, func=my_func)

        ctx = Context({"input": 5})
        stage.execute(ctx)

        assert ctx["result"] == 10

    def test_functional_stage_with_closure(self):
        """Test functional stage with closure."""
        multiplier = 3

        def multiply(ctx):
            ctx["output"] = ctx["value"] * multiplier

        stage = FunctionalStage(
            config=StageConfig(name="multiply", inputs=["value"], outputs=["output"]),
            func=multiply,
        )

        ctx = Context({"value": 7})
        stage.execute(ctx)

        assert ctx["output"] == 21
