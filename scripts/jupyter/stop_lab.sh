#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage: stop_lab.sh [TIMEOUT_SECONDS]

Stop running JupyterLab process(es), waiting up to TIMEOUT_SECONDS
before forcing termination.
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

PATTERN="jupyter-lab|jupyter lab|jupyterlab"
TIMEOUT_SECONDS="${1:-10}"
if ! [[ "${TIMEOUT_SECONDS}" =~ ^[0-9]+$ ]]; then
  echo "[stop_lab] TIMEOUT_SECONDS must be a number. Got: ${TIMEOUT_SECONDS}" >&2
  exit 1
fi

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
