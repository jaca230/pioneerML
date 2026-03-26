# PIONEER ML

PIONEER ML is the framework layer. Model-specific implementations are provided through plugins.

## Plugin-first workflow

- Core framework lives in this repo (`src/pioneerml`).
- Concrete models/loaders/writers/pipeline configs live in plugin repos.
- Current reference plugin repo:
  - https://github.com/jaca230/pioneerML_base_plugin

## Quick start

Clone with submodules:

```bash
git clone --recurse-submodules https://github.com/jaca230/pioneerML.git
cd pioneerML
```

Docker (recommended):

```bash
./scripts/docker/build.sh
./scripts/docker/run.sh
```

Example using a static container, gpu, and port 8888 exposed for notebooks:
```bash
./scripts/docker/run.sh --static --gpu -p 8888:8888
```

Local environment (optional):

```bash
./scripts/env/setup_uv_conda.sh
conda activate pioneerml
```

## Where to start

- Example notebooks: in the base plugin submodule under
  - `plugins/base_plugin/src/pioneerml_base_plugin/...`
