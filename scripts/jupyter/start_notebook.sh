#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage: start_notebook.sh [PORT] [HOST]

Start Jupyter Notebook in the background.

Arguments:
  PORT  TCP port to bind (default: 8888)
  HOST  Interface/IP to bind (default: 0.0.0.0)

Environment:
  JUPYTER_LOG_DIR  Directory for log files
                   (default: <repo>/.runtime/jupyter)
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

if [[ $# -gt 2 ]]; then
  echo "[start_notebook] Too many arguments." >&2
  usage >&2
  exit 1
fi

PORT="${1:-8888}"
HOST="${2:-0.0.0.0}"
if ! [[ "${PORT}" =~ ^[0-9]+$ ]]; then
  echo "[start_notebook] PORT must be a number. Got: ${PORT}" >&2
  exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
DEFAULT_LOG_DIR="${REPO_ROOT}/.runtime/jupyter"
LOG_DIR="${JUPYTER_LOG_DIR:-${DEFAULT_LOG_DIR}}"
LOG_FILE="${LOG_DIR}/jupyter_notebook_${PORT}.log"

mkdir -p "${LOG_DIR}"

echo "[start_notebook] Starting Jupyter Notebook on ${HOST}:${PORT}"
echo "[start_notebook] Log: ${LOG_FILE}"

OLD_PIDS=$(pgrep -f "jupyter-notebook|jupyter notebook|jupyter-notebook" || true)
if [[ -n "${OLD_PIDS}" ]]; then
  echo "[start_notebook] Stopping existing Jupyter Notebook: ${OLD_PIDS}"
  kill ${OLD_PIDS} || true
fi

nohup jupyter notebook \
  --ip="${HOST}" \
  --port="${PORT}" \
  --no-browser \
  --allow-root \
  --NotebookApp.allow_origin='*' \
  --NotebookApp.token='' \
  --NotebookApp.password='' \
  > "${LOG_FILE}" 2>&1 &

PID=$!
echo "[start_notebook] PID: ${PID}"

for i in {1..10}; do
  if grep -q "http" "${LOG_FILE}"; then
    break
  fi
  sleep 1
done

if grep -a -q "http" "${LOG_FILE}"; then
  echo "[start_notebook] URLs:"
  grep -a -Eo "http://[^ ]+" "${LOG_FILE}" | sed 's/\\x1b\\[[0-9;]*m//g'
else
  echo "[start_notebook] Waiting for Jupyter to emit URLs..."
  tail -n 5 "${LOG_FILE}"
fi
