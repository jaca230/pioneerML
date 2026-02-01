#!/usr/bin/env bash
set -euo pipefail

PORT="${1:-8888}"
HOST="${2:-0.0.0.0}"
LOG_DIR="${JUPYTER_LOG_DIR:-/workspace/.jupyter}"
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
