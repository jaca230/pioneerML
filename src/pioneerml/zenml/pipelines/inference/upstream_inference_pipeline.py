"""
ZenML pipeline wrapping the upstream inference flow:
1) Load hits/info NPY batches into GraphRecords.
2) Group by event.
3) Run classifier + endpoint regressor to attach group_probs and endpoint quantiles.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Dict, List, Optional

import torch
from zenml import pipeline, step

from pioneerml.data import load_hits_and_info, GraphRecord
from pioneerml.pipelines.upstream import UpstreamPipeline
from pioneerml.zenml.utils import detect_available_accelerator


@step
def load_hits_info_step(
    hits_pattern: str,
    info_pattern: str,
    max_files: Optional[int] = None,
    limit_groups: Optional[int] = None,
    min_hits: int = 2,
    include_hit_labels: bool = False,
) -> List[GraphRecord]:
    return load_hits_and_info(
        hits_pattern=hits_pattern,
        info_pattern=info_pattern,
        max_files=max_files,
        limit_groups=limit_groups,
        min_hits=min_hits,
        include_hit_labels=include_hit_labels,
        verbose=True,
    )


@step
def group_by_event_step(records: List[GraphRecord]) -> List[List[GraphRecord]]:
    grouped: Dict[int, List[GraphRecord]] = defaultdict(list)
    for rec in records:
        grouped[int(rec.event_id)] += [rec]
    return list(grouped.values())


@step(enable_cache=False)
def run_upstream_step(
    events: List[List[GraphRecord]],
    classifier_path: str,
    endpoint_path: str,
    classifier_config: Optional[Dict[str, Any]] = None,
    endpoint_config: Optional[Dict[str, Any]] = None,
) -> List[List[GraphRecord]]:
    accelerator, devices = detect_available_accelerator()
    device = torch.device(devices[0]) if devices else torch.device("cpu")
    pipe = UpstreamPipeline(device=device)
    pipe.load_models(classifier_path, endpoint_path, classifier_config, endpoint_config)
    pipe.process_unmixed_events(events, batch_size=200)
    return events


@pipeline(enable_cache=False)
def upstream_inference_pipeline(
    hits_pattern: str,
    info_pattern: str,
    classifier_path: str,
    endpoint_path: str,
    max_files: Optional[int] = None,
    limit_groups: Optional[int] = None,
    min_hits: int = 2,
    include_hit_labels: bool = False,
    classifier_config: Optional[Dict[str, Any]] = None,
    endpoint_config: Optional[Dict[str, Any]] = None,
):
    records = load_hits_info_step(
        hits_pattern=hits_pattern,
        info_pattern=info_pattern,
        max_files=max_files,
        limit_groups=limit_groups,
        min_hits=min_hits,
        include_hit_labels=include_hit_labels,
    )
    events = group_by_event_step(records)
    _ = run_upstream_step(
        events,
        classifier_path=classifier_path,
        endpoint_path=endpoint_path,
        classifier_config=classifier_config,
        endpoint_config=endpoint_config,
    )
