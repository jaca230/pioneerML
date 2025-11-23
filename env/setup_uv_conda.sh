#!/usr/bin/env bash
set -euo pipefail

# This script bootstraps a minimal conda env with Python, installs uv,
# and uses uv to install project dependencies from requirements.txt.
#
# Usage:
#   ./env/setup_uv_conda.sh

ENV_NAME="pioneerml-uv"
REQ_FILE="requirements.txt"

if ! command -v conda >/dev/null 2>&1; then
  echo "conda is not available on PATH. Please install Miniconda/Anaconda first." >&2
  exit 1
fi

if conda env list | grep -q "^${ENV_NAME}[[:space:]]"; then
  echo "Updating existing conda env: ${ENV_NAME}"
else
  echo "Creating conda env: ${ENV_NAME} (python=3.11)"
  conda create -y -n "${ENV_NAME}" python=3.11
fi

echo "Activating env..."
eval "$(conda shell.bash hook)"
conda activate "${ENV_NAME}"

echo "Installing uv inside env..."
python -m pip install -U uv

echo "Installing dependencies from ${REQ_FILE} using uv..."
uv pip install -r "${REQ_FILE}"

echo "Done. Activate with: conda activate ${ENV_NAME}"
