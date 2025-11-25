"""
Python version of the ZenML quickstart notebook for debugging.
"""

from zenml.client import Client

from pioneerml.zenml.pipelines import quickstart_pipeline


def main():
    # Initialize local ZenML repo (no server needed)
    Client().initialize()
    # Run the pipeline
    run = quickstart_pipeline()
    print("Pipeline run completed:", run)


if __name__ == "__main__":
    main()
