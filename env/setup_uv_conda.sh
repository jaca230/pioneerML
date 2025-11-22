#!/usr/bin/env bash
set -euo pipefail

# This script bootstraps a minimal conda env with Python, installs uv,
# and uses uv to install project dependencies from env/requirements.txt
# (or env/requirements-dev.txt when --dev is passed).
#
# Usage:
#   ./env/setup_uv_conda.sh          # core deps
#   ./env/setup_uv_conda.sh --dev    # core + dev/test deps

ENV_NAME="pioneerml-uv"
REQ_FILE="env/requirements.txt"

if [ "${1:-}" = "--dev" ]; then
  REQ_FILE="env/requirements-dev.txt"
fi

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
