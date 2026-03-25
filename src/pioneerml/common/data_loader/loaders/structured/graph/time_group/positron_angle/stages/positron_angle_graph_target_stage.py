from __future__ import annotations

from pioneerml.common.data_loader.loaders.stage.stages import GraphTargetStage


class PositronAngleGraphTargetStage(GraphTargetStage):
    """Build base positron-angle graph targets `[px, py, pz]` before quantile expansion."""

    name = "build_targets"
    requires = ("layout", "local_gid", "row_ids_graph")
    provides = ("y_graph",)

    def __init__(
        self,
        *,
        source_state_key: str = "features_in",
    ) -> None:
        super().__init__(
            target_specs=(
                ("positron_px", 0),
                ("positron_py", 1),
                ("positron_pz", 2),
            ),
            num_classes=3,
            source_state_key=source_state_key,
        )

