"""
ZenML pipeline wrapping the downstream inference flow on mixed events:
1) Load a MixedEventDataset from disk.
2) Run splitter -> pion stop -> positron angle models.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import torch
from zenml import pipeline, step

from pioneerml.data.event_mixer import MixedEventDataset, EventContainer
from pioneerml.pipelines.downstream import DownstreamPipeline
from pioneerml.zenml.utils import detect_available_accelerator


@step
def load_mixed_dataset_step(path: str) -> MixedEventDataset:
    return MixedEventDataset(path)


@step(enable_cache=False)
def run_downstream_step(
    dataset: MixedEventDataset,
    splitter_path: str,
    pi_stop_path: str,
    pos_angle_path: str,
) -> List[EventContainer]:
    accelerator, devices = detect_available_accelerator()
    device = torch.device(devices[0]) if devices else torch.device("cpu")
    pipe = DownstreamPipeline(device=device)
    pipe.load_models(splitter_path, pi_stop_path, pos_angle_path)
    outputs: List[EventContainer] = []
    for event in dataset:
        outputs.append(pipe.process_event(event))
    return outputs


@pipeline(enable_cache=False)
def downstream_inference_pipeline(
    mixed_dataset_path: str,
    splitter_path: str,
    pi_stop_path: str,
    pos_angle_path: str,
):
    ds = load_mixed_dataset_step(mixed_dataset_path)
    _ = run_downstream_step(ds, splitter_path, pi_stop_path, pos_angle_path)
