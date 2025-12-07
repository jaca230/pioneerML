"""
Timestamp utilities.
"""


def timestamp_now() -> str:
    """Return a filesystem-friendly UTC timestamp."""
    from datetime import datetime, timezone

    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")


