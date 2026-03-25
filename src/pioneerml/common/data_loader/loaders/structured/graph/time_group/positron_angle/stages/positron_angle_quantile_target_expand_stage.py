from __future__ import annotations

from pioneerml.common.data_loader.loaders.structured.graph.time_group.pion_stop.stages.pion_stop_quantile_target_expand_stage import (
    PionStopQuantileTargetExpandStage,
)


class PositronAngleQuantileTargetExpandStage(PionStopQuantileTargetExpandStage):
    """Expand base positron-angle targets `[3] -> [9]` by quantile-slot repetition."""

