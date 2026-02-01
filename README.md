# PIONEER ML

Lightweight setup guide for the PIONEER ML codebase.

## Quick start

```bash
git clone git@github.com:jaca230/pioneerML.git
cd pioneerML

# easiest: conda + uv helper
./scripts/env/setup_uv_conda.sh
conda activate pioneerml

# fallback (not recommended): manual venv
# python -m venv .venv
# source .venv/bin/activate
# pip install -e .
```


## Notebooks
Training / validation / data-generation notebooks live under `notebooks/` (training, validation, data_generation subfolders). Run them after activating the env above.
