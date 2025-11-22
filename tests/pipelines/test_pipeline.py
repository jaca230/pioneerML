"""
Tests for Pipeline executor.
"""

import pytest
from pioneerml.pipelines import Pipeline, Stage, StageConfig, FunctionalStage, Context


class TestPipeline:
    """Tests for Pipeline class."""

    def test_simple_pipeline(self):
        """Test simple linear pipeline."""

        def load(ctx):
            ctx["data"] = [1, 2, 3]

        def process(ctx):
            ctx["result"] = sum(ctx["data"])

        pipeline = Pipeline(
            [
                FunctionalStage(config=StageConfig(name="load", outputs=["data"]), func=load),
                FunctionalStage(
                    config=StageConfig(name="process", inputs=["data"], outputs=["result"]),
                    func=process,
                ),
            ]
        )

        ctx = pipeline.run()

        assert ctx["data"] == [1, 2, 3]
        assert ctx["result"] == 6

    def test_pipeline_with_initial_context(self):
        """Test pipeline with pre-populated context."""

        def process(ctx):
            ctx["output"] = ctx["input"] * 2

        pipeline = Pipeline(
            [
                FunctionalStage(
                    config=StageConfig(name="process", inputs=["input"], outputs=["output"]),
                    func=process,
                )
            ]
        )

        ctx = Context({"input": 10})
        result_ctx = pipeline.run(ctx)

        assert result_ctx["output"] == 20

    def test_execution_order(self):
        """Test that stages execute in dependency order."""
        execution_order = []

        def stage_a(ctx):
            execution_order.append("A")
            ctx["a"] = 1

        def stage_b(ctx):
            execution_order.append("B")
            ctx["b"] = ctx["a"] + 1

        def stage_c(ctx):
            execution_order.append("C")
            ctx["c"] = ctx["b"] + 1

        # Define in reverse order - should still execute correctly
        pipeline = Pipeline(
            [
                FunctionalStage(
                    config=StageConfig(name="c", inputs=["b"], outputs=["c"]),
                    func=stage_c,
                ),
                FunctionalStage(
                    config=StageConfig(name="b", inputs=["a"], outputs=["b"]),
                    func=stage_b,
                ),
                FunctionalStage(
                    config=StageConfig(name="a", outputs=["a"]),
                    func=stage_a,
                ),
            ]
        )

        ctx = pipeline.run()

        assert execution_order == ["A", "B", "C"]
        assert ctx["c"] == 3

    def test_branching_pipeline(self):
        """Test pipeline with branches that converge."""

        def start(ctx):
            ctx["data"] = 10

        def branch_a(ctx):
            ctx["result_a"] = ctx["data"] * 2

        def branch_b(ctx):
            ctx["result_b"] = ctx["data"] + 5

        def merge(ctx):
            ctx["final"] = ctx["result_a"] + ctx["result_b"]

        pipeline = Pipeline(
            [
                FunctionalStage(config=StageConfig(name="start", outputs=["data"]), func=start),
                FunctionalStage(
                    config=StageConfig(name="branch_a", inputs=["data"], outputs=["result_a"]),
                    func=branch_a,
                ),
                FunctionalStage(
                    config=StageConfig(name="branch_b", inputs=["data"], outputs=["result_b"]),
                    func=branch_b,
                ),
                FunctionalStage(
                    config=StageConfig(
                        name="merge",
                        inputs=["result_a", "result_b"],
                        outputs=["final"],
                    ),
                    func=merge,
                ),
            ]
        )

        ctx = pipeline.run()

        assert ctx["result_a"] == 20
        assert ctx["result_b"] == 15
        assert ctx["final"] == 35

    def test_circular_dependency_detection(self):
        """Test that circular dependencies are detected."""

        def noop(ctx):
            pass

        with pytest.raises(ValueError, match="Circular dependencies"):
            Pipeline(
                [
                    FunctionalStage(
                        config=StageConfig(name="a", inputs=["c"], outputs=["a"]),
                        func=noop,
                    ),
                    FunctionalStage(
                        config=StageConfig(name="b", inputs=["a"], outputs=["b"]),
                        func=noop,
                    ),
                    FunctionalStage(
                        config=StageConfig(name="c", inputs=["b"], outputs=["c"]),
                        func=noop,
                    ),
                ],
                validate=True,
            )

    def test_duplicate_stage_names(self):
        """Test that duplicate stage names are detected."""

        def noop(ctx):
            pass

        with pytest.raises(ValueError, match="Duplicate stage names"):
            Pipeline(
                [
                    FunctionalStage(config=StageConfig(name="same", outputs=["a"]), func=noop),
                    FunctionalStage(config=StageConfig(name="same", outputs=["b"]), func=noop),
                ]
            )

    def test_skip_stages(self):
        """Test skipping stages."""
        execution_log = []

        def stage_a(ctx):
            execution_log.append("A")
            ctx["a"] = 1

        def stage_b(ctx):
            execution_log.append("B")
            ctx["b"] = 2

        def stage_c(ctx):
            execution_log.append("C")
            ctx["c"] = 3

        pipeline = Pipeline(
            [
                FunctionalStage(config=StageConfig(name="a", outputs=["a"]), func=stage_a),
                FunctionalStage(config=StageConfig(name="b", outputs=["b"]), func=stage_b),
                FunctionalStage(config=StageConfig(name="c", outputs=["c"]), func=stage_c),
            ]
        )

        ctx = pipeline.run(skip_stages={"b"})

        assert execution_log == ["A", "C"]
        assert "a" in ctx
        assert "b" not in ctx
        assert "c" in ctx

    def test_disabled_stage(self):
        """Test that disabled stages are skipped."""
        execution_log = []

        def stage_func(ctx):
            execution_log.append("executed")

        config = StageConfig(name="test", enabled=False)
        stage = FunctionalStage(config, lambda ctx: stage_func(ctx))

        pipeline = Pipeline([stage])
        pipeline.run()

        assert len(execution_log) == 0

    def test_enable_disable_stage(self):
        """Test enabling and disabling stages."""

        def noop(ctx):
            pass

        pipeline = Pipeline(
            [FunctionalStage(config=StageConfig(name="test", outputs=["a"]), func=noop)]
        )

        pipeline.disable_stage("test")
        assert not pipeline.get_stage("test").enabled

        pipeline.enable_stage("test")
        assert pipeline.get_stage("test").enabled

    def test_visualize(self):
        """Test pipeline visualization."""

        def noop(ctx):
            pass

        pipeline = Pipeline(
            [
                FunctionalStage(config=StageConfig(name="load", outputs=["data"]), func=noop),
                FunctionalStage(
                    config=StageConfig(name="process", inputs=["data"], outputs=["result"]),
                    func=noop,
                ),
            ],
            name="test_pipeline",
        )

        viz = pipeline.visualize()

        assert "test_pipeline" in viz
        assert "load" in viz
        assert "process" in viz
        assert "data" in viz

    def test_get_stage(self):
        """Test getting stage by name."""

        def noop(ctx):
            pass

        stage = FunctionalStage(config=StageConfig(name="test", outputs=["a"]), func=noop)
        pipeline = Pipeline([stage])

        retrieved = pipeline.get_stage("test")
        assert retrieved is stage
        assert pipeline.get_stage("nonexistent") is None
