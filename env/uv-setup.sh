#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

ENV_DIR=".venv"         # directory on disk
ENV_PROMPT="pioneerml"  # name shown in the shell prompt

if ! command -v uv >/dev/null 2>&1; then
  echo "uv is not installed. Install via: curl -LsSf https://astral.sh/uv/install.sh | sh" >&2
  exit 1
fi

# Create environment only if missing
if [ ! -d "$ENV_DIR" ]; then
  echo "Creating $ENV_DIR with prompt '$ENV_PROMPT'..."
  uv venv "$ENV_DIR" --prompt "$ENV_PROMPT"
fi

source "$ENV_DIR/bin/activate"

INSTALL_FILE="$REPO_ROOT/requirements.txt"

echo "Installing dependencies from $INSTALL_FILE using uv..."
uv pip install -r "$INSTALL_FILE"

# Install package in editable mode for clean imports (pioneerml importable without PYTHONPATH hacks)
uv pip install -e "$REPO_ROOT"

echo "Registering Jupyter kernel 'pioneerml' for this env..."
python -m ipykernel install --user --name "${ENV_PROMPT}" --display-name "Python (${ENV_PROMPT})" >/dev/null 2>&1 || true

echo "Done. Activate with: source $ENV_DIR/bin/activate"
