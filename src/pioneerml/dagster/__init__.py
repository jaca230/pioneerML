"""
Dagster definitions for PIONEER ML.
"""

from dagster import Definitions

from pioneerml.dagster.jobs import dagster_train_job, dagster_train_and_infer_job

__all__ = ["defs"]

defs = Definitions(jobs=[dagster_train_job, dagster_train_and_infer_job])
