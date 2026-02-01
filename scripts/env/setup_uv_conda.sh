#!/usr/bin/env bash
set -euo pipefail

# This script bootstraps a minimal conda env with Python, installs uv,
# and uses uv to install project dependencies from requirements.txt.
#
# Usage:
#   ./scripts/env/setup_uv_conda.sh

ENV_NAME="pioneerml"
REQ_FILE="requirements.txt"
PYTHON_VERSION="${PYTHON_VERSION:-3.11}"

if ! command -v conda >/dev/null 2>&1; then
  echo "conda is not available on PATH. Please install Miniconda/Anaconda first." >&2
  exit 1
fi

if conda env list | grep -q "^${ENV_NAME}[[:space:]]"; then
  echo "Updating existing conda env: ${ENV_NAME}"
else
  echo "Creating conda env: ${ENV_NAME} (python=${PYTHON_VERSION})"
  conda create -y -n "${ENV_NAME}" "python=${PYTHON_VERSION}"
fi

echo "Activating env..."
eval "$(conda shell.bash hook)"
conda activate "${ENV_NAME}"

echo "Installing uv inside env..."
python -m pip install -U uv

echo "Installing torch nightly (CUDA 12.8) using uv..."
uv pip install --prerelease=allow \
  --index-url "https://download.pytorch.org/whl/nightly/cu128" \
  "torch==2.11.0.dev20260131+cu128"

echo "Installing dependencies from ${REQ_FILE} using uv..."
uv pip install -r "${REQ_FILE}"

# Install package in editable mode for clean imports (pioneerml importable without PYTHONPATH hacks)
uv pip install -e .

# Register kernel for notebooks
echo "Registering Jupyter kernel 'pioneerml' for this env..."
python -m ipykernel install --user --name "${ENV_NAME}" --display-name "Python (${ENV_NAME})" >/dev/null 2>&1 || true

echo "Done. Activate with: conda activate ${ENV_NAME}"
