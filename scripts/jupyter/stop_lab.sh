#!/usr/bin/env bash
set -euo pipefail

PATTERN="jupyter-lab|jupyter lab|jupyterlab"
TIMEOUT_SECONDS="${1:-10}"

echo "[stop_lab] Looking for running JupyterLab processes..."
PIDS="$(pgrep -f "${PATTERN}" || true)"

if [[ -z "${PIDS}" ]]; then
  echo "[stop_lab] No running JupyterLab process found."
  exit 0
fi

echo "[stop_lab] Stopping JupyterLab PID(s): ${PIDS}"
kill ${PIDS} || true

for ((i=0; i<TIMEOUT_SECONDS; i++)); do
  sleep 1
  STILL_RUNNING="$(pgrep -f "${PATTERN}" || true)"
  if [[ -z "${STILL_RUNNING}" ]]; then
    echo "[stop_lab] JupyterLab stopped."
    exit 0
  fi
done

STILL_RUNNING="$(pgrep -f "${PATTERN}" || true)"
if [[ -n "${STILL_RUNNING}" ]]; then
  echo "[stop_lab] Forcing stop for PID(s): ${STILL_RUNNING}"
  kill -9 ${STILL_RUNNING} || true
fi

echo "[stop_lab] Done."

