"""
Example demonstrating the pipeline framework.

This shows how to build a simple ML pipeline using the DAG-based system.
"""

import torch
from torch_geometric.data import Data, DataLoader

from pioneerml.pipelines import Pipeline, Stage, StageConfig, Context, FunctionalStage
from pioneerml.models import GroupClassifier


# Example 1: Simple pipeline with functional stages
def example_simple_pipeline():
    """Demonstrate pipeline with simple functional stages."""

    print("=" * 60)
    print("Example 1: Simple Pipeline with Functional Stages")
    print("=" * 60)

    # Define simple functions
    def load_data(ctx):
        """Create dummy dataset."""
        print("Loading data...")
        ctx["raw_data"] = list(range(10))

    def preprocess(ctx):
        """Double the numbers."""
        print("Preprocessing...")
        ctx["processed_data"] = [x * 2 for x in ctx["raw_data"]]

    def analyze(ctx):
        """Compute statistics."""
        print("Analyzing...")
        data = ctx["processed_data"]
        ctx["stats"] = {"mean": sum(data) / len(data), "max": max(data), "min": min(data)}

    # Create pipeline
    pipeline = Pipeline(
        [
            FunctionalStage(
                config=StageConfig(name="load", outputs=["raw_data"]),
                func=load_data,
            ),
            FunctionalStage(
                config=StageConfig(
                    name="preprocess",
                    inputs=["raw_data"],
                    outputs=["processed_data"],
                ),
                func=preprocess,
            ),
            FunctionalStage(
                config=StageConfig(
                    name="analyze",
                    inputs=["processed_data"],
                    outputs=["stats"],
                ),
                func=analyze,
            ),
        ],
        name="simple_pipeline",
    )

    # Visualize
    print("\n" + pipeline.visualize())

    # Run
    print("\n" + "Running pipeline...")
    ctx = pipeline.run()

    print("\nResults:")
    print(f"Raw data: {ctx['raw_data']}")
    print(f"Processed data: {ctx['processed_data']}")
    print(f"Statistics: {ctx['stats']}")


# Example 2: Custom stage classes
class CreateDatasetStage(Stage):
    """Create a dummy torch_geometric dataset."""

    def execute(self, context):
        """Create dataset."""
        print(f"[{self.name}] Creating dataset...")

        # Create dummy graphs
        dataset = []
        for i in range(20):
            x = torch.randn(10, 5)  # 10 nodes, 5 features
            edge_index = torch.randint(0, 10, (2, 30))
            edge_attr = torch.randn(30, 4)
            y = torch.randint(0, 2, (3,)).float()  # 3-class multi-label

            dataset.append(Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y))

        context["dataset"] = dataset


class CreateDataLoadersStage(Stage):
    """Split dataset and create data loaders."""

    def execute(self, context):
        """Create loaders."""
        print(f"[{self.name}] Creating data loaders...")

        dataset = context["dataset"]
        batch_size = self.config.params.get("batch_size", 8)

        # Simple split
        train_size = int(0.8 * len(dataset))
        train_dataset = dataset[:train_size]
        val_dataset = dataset[train_size:]

        context["train_loader"] = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        context["val_loader"] = DataLoader(val_dataset, batch_size=batch_size)


class SimpleTrainStage(Stage):
    """Train a model (simplified)."""

    def execute(self, context):
        """Train model."""
        print(f"[{self.name}] Training model...")

        model = GroupClassifier(hidden=64, num_blocks=2, num_classes=3)
        train_loader = context["train_loader"]

        # Simple training (just 1 epoch for demo)
        model.train()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        criterion = torch.nn.BCEWithLogitsLoss()

        total_loss = 0
        for batch in train_loader:
            optimizer.zero_grad()
            out = model(batch)
            loss = criterion(out, batch.y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        context["model"] = model
        context["train_loss"] = total_loss / len(train_loader)


def example_ml_pipeline():
    """Demonstrate ML pipeline with custom stages."""

    print("\n" + "=" * 60)
    print("Example 2: ML Pipeline with Custom Stages")
    print("=" * 60)

    pipeline = Pipeline(
        [
            CreateDatasetStage(
                config=StageConfig(
                    name="create_dataset",
                    outputs=["dataset"],
                )
            ),
            CreateDataLoadersStage(
                config=StageConfig(
                    name="create_loaders",
                    inputs=["dataset"],
                    outputs=["train_loader", "val_loader"],
                    params={"batch_size": 4},
                )
            ),
            SimpleTrainStage(
                config=StageConfig(
                    name="train",
                    inputs=["train_loader"],
                    outputs=["model", "train_loss"],
                )
            ),
        ],
        name="ml_pipeline",
    )

    # Visualize
    print("\n" + pipeline.visualize())

    # Run
    print("\n" + "Running pipeline...")
    ctx = pipeline.run()

    print(f"\nTraining completed!")
    print(f"Final loss: {ctx['train_loss']:.4f}")
    print(f"Model: {ctx['model']}")


# Example 3: Branching pipeline
def example_branching_pipeline():
    """Demonstrate pipeline with branches."""

    print("\n" + "=" * 60)
    print("Example 3: Branching Pipeline")
    print("=" * 60)

    def load(ctx):
        ctx["data"] = list(range(10))

    def path_a(ctx):
        """Process data one way."""
        ctx["result_a"] = sum(ctx["data"])

    def path_b(ctx):
        """Process data another way."""
        ctx["result_b"] = max(ctx["data"]) - min(ctx["data"])

    def combine(ctx):
        """Combine results from both paths."""
        ctx["final"] = ctx["result_a"] + ctx["result_b"]

    pipeline = Pipeline(
        [
            FunctionalStage(config=StageConfig(name="load", outputs=["data"]), func=load),
            FunctionalStage(
                config=StageConfig(name="path_a", inputs=["data"], outputs=["result_a"]),
                func=path_a,
            ),
            FunctionalStage(
                config=StageConfig(name="path_b", inputs=["data"], outputs=["result_b"]),
                func=path_b,
            ),
            FunctionalStage(
                config=StageConfig(
                    name="combine",
                    inputs=["result_a", "result_b"],
                    outputs=["final"],
                ),
                func=combine,
            ),
        ],
        name="branching_pipeline",
    )

    print("\n" + pipeline.visualize())
    print("\nNote: Stages 'path_a' and 'path_b' can run in parallel (future feature)")

    ctx = pipeline.run()

    print(f"\nResults:")
    print(f"Result A (sum): {ctx['result_a']}")
    print(f"Result B (range): {ctx['result_b']}")
    print(f"Final (combined): {ctx['final']}")


if __name__ == "__main__":
    example_simple_pipeline()
    example_ml_pipeline()
    example_branching_pipeline()

    print("\n" + "=" * 60)
    print("All examples completed successfully!")
    print("=" * 60)
