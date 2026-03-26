from __future__ import annotations

from pioneerml.staged_runtime import BaseDiagnostics


class WriterDiagnostics(BaseDiagnostics):
    def __init__(self, *, writer_kind: str = "writer") -> None:
        super().__init__(runtime_kind=str(writer_kind))
        self.writer_kind = str(writer_kind)

    def summary(self) -> dict:
        out = super().summary()
        out["writer_kind"] = self.writer_kind
        return out
