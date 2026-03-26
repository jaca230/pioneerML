from __future__ import annotations

from abc import abstractmethod
from collections.abc import MutableMapping
from typing import Any, TYPE_CHECKING

from pioneerml.data_writer.stage.stages.base_stage import BaseWriterStage

if TYPE_CHECKING:
    from ..time_group_graph_data_writer import TimeGroupGraphDataWriter


class BaseTimeGroupWriterStage(BaseWriterStage):
    """Writer stage base with enforced TimeGroupGraphDataWriter owner type."""

    def run_writer(self, *, state: MutableMapping[str, Any], owner) -> None:
        from ..time_group_graph_data_writer import TimeGroupGraphDataWriter

        if not isinstance(owner, TimeGroupGraphDataWriter):
            raise TypeError(
                f"{self.__class__.__name__} expected owner type TimeGroupGraphDataWriter, "
                f"got {type(owner).__name__}."
            )
        self.run_time_group_writer(state=state, owner=owner)

    @abstractmethod
    def run_time_group_writer(self, *, state: MutableMapping[str, Any], owner: "TimeGroupGraphDataWriter") -> None:
        raise NotImplementedError
