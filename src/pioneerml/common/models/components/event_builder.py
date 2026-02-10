"""Legacy compatibility alias for the renamed EventSplitter model."""

from __future__ import annotations

from pioneerml.common.models.components.event_splitter import EventSplitter


class EventBuilder(EventSplitter):
    """Backward-compatible alias for `EventSplitter`."""
