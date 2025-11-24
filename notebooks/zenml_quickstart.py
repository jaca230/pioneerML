"""
Python version of the ZenML quickstart notebook for debugging.
"""

from zenml.client import Client

from zenml_pipelines.training_pipeline import zenml_training_pipeline


def main():
    # Initialize local ZenML repo (no server needed)
    Client().initialize()
    # Run the pipeline
    run = zenml_training_pipeline()
    print("Pipeline run completed:", run)


if __name__ == "__main__":
    main()
