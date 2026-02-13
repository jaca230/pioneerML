class BaseDeriver:
    """Base interface for Python derivers."""

    def derive_row(self, row: dict) -> dict:
        raise NotImplementedError
