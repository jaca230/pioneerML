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
        hpo_output = self.hpo_step(dataset=dataset, pipeline_config=pipeline_config)
        train_output = self.train_step(dataset=dataset, pipeline_config=pipeline_config, hpo_payload=hpo_output)
        metrics = self.evaluate_step(train_payload=train_output, dataset=dataset, pipeline_config=pipeline_config)
        export = self.export_step(
            train_payload=train_output,
            dataset=dataset,
            pipeline_config=pipeline_config,
            hpo_payload=hpo_output,
            metrics=metrics,
        )
        module = train_output
        return module, dataset, metrics, export
