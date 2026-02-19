from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path


class BaseEvaluator(ABC):
    def concise_history(self, history: list[float], *, max_points: int = 20) -> tuple[list[float], int]:
        if len(history) <= max_points:
            return list(history), len(history)
        return list(history[-max_points:]), len(history)

    def resolve_plot_path(self, config: dict | None) -> str | None:
        if not config:
            return None
        if config.get("plot_path"):
            return str(config["plot_path"])
        if config.get("plot_dir"):
            plot_dir = Path(str(config["plot_dir"]))
            plot_dir.mkdir(parents=True, exist_ok=True)
            return str(plot_dir / "loss_curves.png")
        return None

    @abstractmethod
    def evaluate(self, **kwargs) -> dict:
        raise NotImplementedError
