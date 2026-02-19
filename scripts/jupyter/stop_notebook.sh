#!/usr/bin/env bash
set -euo pipefail

PATTERN="jupyter-notebook|jupyter notebook"
TIMEOUT_SECONDS="${1:-10}"

echo "[stop_notebook] Looking for running Jupyter Notebook processes..."
PIDS="$(pgrep -f "${PATTERN}" || true)"

if [[ -z "${PIDS}" ]]; then
  echo "[stop_notebook] No running Jupyter Notebook process found."
  exit 0
fi

echo "[stop_notebook] Stopping Jupyter Notebook PID(s): ${PIDS}"
kill ${PIDS} || true

for ((i=0; i<TIMEOUT_SECONDS; i++)); do
  sleep 1
  STILL_RUNNING="$(pgrep -f "${PATTERN}" || true)"
  if [[ -z "${STILL_RUNNING}" ]]; then
    echo "[stop_notebook] Jupyter Notebook stopped."
    exit 0
  fi
done

STILL_RUNNING="$(pgrep -f "${PATTERN}" || true)"
if [[ -n "${STILL_RUNNING}" ]]; then
  echo "[stop_notebook] Forcing stop for PID(s): ${STILL_RUNNING}"
  kill -9 ${STILL_RUNNING} || true
fi

echo "[stop_notebook] Done."

