# PIONEER ML

PIONEER ML is an ML framework for the PIONEER experiment focused on scalable graph-model training and inference with streamed data/predictions.

![Python](https://img.shields.io/badge/Python-3.9%2B-3776AB?logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-2.x-EE4C2C?logo=pytorch&logoColor=white)
![PyG](https://img.shields.io/badge/PyTorch%20Geometric-2.4%2B-3C2179)
![Lightning](https://img.shields.io/badge/PyTorch%20Lightning-2.x-792EE5)
![ZenML](https://img.shields.io/badge/ZenML-0.92-4B9CD3)
![Optuna](https://img.shields.io/badge/Optuna-3.5%2B-4C4CFF)

## What this repo provides

1. Streaming-first pipelines for training and inference.
2. Multiple model stages (classification, splitting, regression).
3. ZenML-based pipeline orchestration.
4. Notebook workflows for training, inference, and validation.

## Quick start (Conda, recommended)

```bash
git clone git@github.com:jaca230/pioneerML.git
cd pioneerML
./scripts/env/setup_uv_conda.sh
conda activate pioneerml
```

Notes:

1. The setup script creates a conda environment, installs dependencies, and registers a Jupyter kernel named `pioneerml`.
2. Script reference: `scripts/env/setup_uv_conda.sh`.
3. Alternative venv flow: `scripts/env/setup_uv_venv.sh`.

## Quick start (Docker)

Build image:

```bash
./scripts/docker/build.sh
```

Run container:

```bash
./scripts/docker/run.sh
```

Useful options:

1. `--static` to reuse a persistent container.
2. `-p 8888:8888` to expose Jupyter.
3. `--gpu` or `--gpus all` for GPU access.

Script reference:

1. `scripts/docker/build.sh`
2. `scripts/docker/run.sh`

## Start Jupyter

JupyterLab:

```bash
./scripts/jupyter/start_lab.sh 8888
```

Classic Notebook:

```bash
./scripts/jupyter/start_notebook.sh 8888
```

Stop helpers:

1. `scripts/jupyter/stop_lab.sh`
2. `scripts/jupyter/stop_notebook.sh`

## Recommended notebook path

Start here:

1. `notebooks/examples/tutorials/00_quickstart.ipynb`
2. `notebooks/examples/tutorials/01_building_zenml_pipelines.ipynb`

Then run one full model flow:

1. Train: `notebooks/training/group_classification.ipynb`
2. Inference: `notebooks/inference/group_classification.ipynb`
3. Validation: `notebooks/validation/group_classification.ipynb`

Additional model notebooks are under:

1. `notebooks/training/`
2. `notebooks/inference/`
3. `notebooks/validation/`

## Project layout

1. `src/pioneerml/`: core library, models, loaders, pipelines.
2. `notebooks/`: interactive workflows.
3. `scripts/`: environment, docker, and jupyter helpers.
4. `data/`: local parquet data and generated prediction outputs.
