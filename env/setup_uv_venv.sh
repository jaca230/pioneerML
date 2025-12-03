#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

ENV_DIR=".venv"
ENV_PROMPT="pioneerml"

if ! command -v uv >/dev/null 2>&1; then
  echo "uv is not installed. Install via: curl -LsSf https://astral.sh/uv/install.sh | sh" >&2
  exit 1
fi

# Create env if missing
if [ ! -d "$ENV_DIR" ]; then
  echo "Creating $ENV_DIR with prompt '$ENV_PROMPT'..."
  uv venv "$ENV_DIR" --prompt "$ENV_PROMPT"
fi

source "$ENV_DIR/bin/activate"

INSTALL_FILE="$REPO_ROOT/requirements.txt"
echo "Installing dependencies from $INSTALL_FILE using uv..."
uv pip install -r "$INSTALL_FILE"

# Install package in editable mode
uv pip install -e "$REPO_ROOT"

# Ensure ipykernel exists in this env
python -m pip install -U ipykernel >/dev/null

# Register Jupyter kernel automatically if not already present
if ! jupyter kernelspec list 2>/dev/null | grep -q "^${ENV_PROMPT}\b"; then
  echo "Registering Jupyter kernel '$ENV_PROMPT'..."
  python -m ipykernel install --user \
    --name "${ENV_PROMPT}" \
    --display-name "Python (${ENV_PROMPT})"
else
  echo "Kernel '$ENV_PROMPT' already registered."
fi

echo "Done. Activate with: source $ENV_DIR/bin/activate"
