from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable


@dataclass
class TrainingPipelineRunner:
    load_step: Callable[..., Any]
    hpo_step: Callable[..., dict]
    train_step: Callable[..., Any]
    evaluate_step: Callable[..., dict]
    export_step: Callable[..., dict]

    def run(
        self,
        *,
        loader_kwargs: dict,
        pipeline_config: dict | None,
    ):
        dataset = self.load_step(pipeline_config=pipeline_config, **loader_kwargs)
        hpo_params = self.hpo_step(dataset=dataset, pipeline_config=pipeline_config)
        module = self.train_step(dataset=dataset, pipeline_config=pipeline_config, hpo_params=hpo_params)
        metrics = self.evaluate_step(module=module, dataset=dataset, pipeline_config=pipeline_config)
        export = self.export_step(
            module=module,
            dataset=dataset,
            pipeline_config=pipeline_config,
            hpo_params=hpo_params,
            metrics=metrics,
        )
        return module, dataset, metrics, export

