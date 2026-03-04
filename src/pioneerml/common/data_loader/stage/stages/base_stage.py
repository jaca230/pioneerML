from __future__ import annotations

from abc import abstractmethod
from collections.abc import MutableMapping
from typing import Any, TYPE_CHECKING

from pioneerml.common.staged_runtime import BaseStage as RuntimeBaseStage

if TYPE_CHECKING:
    from ...structured.structured_loader import StructuredLoader


class BaseStage(RuntimeBaseStage):
    """Loader stage base with enforced StructuredLoader owner type."""

    def run(self, *, state: MutableMapping[str, Any], owner) -> None:
        from ...structured.structured_loader import StructuredLoader

        if not isinstance(owner, StructuredLoader):
            raise TypeError(
                f"{self.__class__.__name__} expected owner type StructuredLoader, got {type(owner).__name__}."
            )
        self.run_loader(state=state, owner=owner)

    @abstractmethod
    def run_loader(self, *, state: MutableMapping[str, Any], owner: "StructuredLoader") -> None:
        raise NotImplementedError


BaseLoaderStage = BaseStage
