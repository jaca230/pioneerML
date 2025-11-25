#!/usr/bin/env python3
"""
Test script for ZenML training pipeline.
This allows for easier debugging than running in a notebook.
"""

from zenml.client import Client
from pioneerml.zenml.pipelines import zenml_training_pipeline

def main():
    print("Initializing ZenML client...")
    try:
        Client().initialize()
    except Exception as e:
        print(f"Repository already initialized: {e}")
        print("Using existing repository...")

    print("Running ZenML training pipeline...")
    run = zenml_training_pipeline()

    print(f"\nPipeline run completed: {run}")
    print(f"Run ID: {run.id}")
    print(f"Run name: {run.name}")
    print(f"Status: {run.status}")

    return run

if __name__ == "__main__":
    main()
