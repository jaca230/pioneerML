from __future__ import annotations

from dataclasses import dataclass
from typing import Callable


@dataclass
class InferencePipelineRunner:
    load_inputs_step: Callable[..., dict]
    load_writer_step: Callable[..., dict]
    load_model_step: Callable[..., dict]
    run_inference_step: Callable[..., dict]

    def run(
        self,
        *,
        loader_kwargs: dict,
        model_path: str | None,
        output_dir: str | None,
        output_path: str | None,
        pipeline_config: dict | None,
    ) -> dict:
        inputs = self.load_inputs_step(pipeline_config=pipeline_config, **loader_kwargs)
        writer_setup = self.load_writer_step(
            inputs=inputs,
            output_dir=output_dir,
            output_path=output_path,
            pipeline_config=pipeline_config,
        )
        model_info = self.load_model_step(model_path=model_path, pipeline_config=pipeline_config)
        outputs = self.run_inference_step(
            model_info=model_info,
            inputs=inputs,
            writer_setup=writer_setup,
            pipeline_config=pipeline_config,
        )
        return outputs
