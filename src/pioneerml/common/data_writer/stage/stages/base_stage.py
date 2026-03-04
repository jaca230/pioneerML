from __future__ import annotations

from abc import abstractmethod
from collections.abc import MutableMapping
from typing import Any, TYPE_CHECKING

from pioneerml.common.staged_runtime import BaseStage

if TYPE_CHECKING:
    from ...structured.structured_data_writer import StructuredDataWriter


class BaseWriterStage(BaseStage):
    """Writer stage base with enforced StructuredDataWriter owner type."""

    def run(self, *, state: MutableMapping[str, Any], owner) -> None:
        from ...structured.structured_data_writer import StructuredDataWriter

        if not isinstance(owner, StructuredDataWriter):
            raise TypeError(
                f"{self.__class__.__name__} expected owner type StructuredDataWriter, got {type(owner).__name__}."
            )
        self.run_writer(state=state, owner=owner)

    @abstractmethod
    def run_writer(self, *, state: MutableMapping[str, Any], owner: "StructuredDataWriter") -> None:
        raise NotImplementedError
