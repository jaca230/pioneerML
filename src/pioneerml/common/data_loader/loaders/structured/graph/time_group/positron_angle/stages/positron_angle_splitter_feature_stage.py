from __future__ import annotations

from pioneerml.common.data_loader.loaders.structured.graph.time_group.pion_stop.stages.pion_stop_splitter_feature_stage import (
    PionStopSplitterFeatureStage,
)


class PositronAngleSplitterFeatureStage(PionStopSplitterFeatureStage):
    """Build per-node splitter priors for positron-angle features."""

