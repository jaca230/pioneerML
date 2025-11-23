#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

if ! command -v uv >/dev/null 2>&1; then
  echo "uv is not installed. Install via: curl -LsSf https://astral.sh/uv/install.sh | sh" >&2
  exit 1
fi

if [ ! -d ".venv" ]; then
  echo "Creating .venv with uv..."
  uv venv .venv
fi

source .venv/bin/activate

INSTALL_FILE="$REPO_ROOT/requirements.txt"

echo "Installing dependencies from $INSTALL_FILE using uv..."
uv pip install -r "$INSTALL_FILE"

echo "Done. Activate with: source .venv/bin/activate"
