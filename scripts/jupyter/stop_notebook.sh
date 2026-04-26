#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage: stop_notebook.sh [--timeout TIMEOUT_SECONDS]
       stop_notebook.sh [TIMEOUT_SECONDS]

Stop running Jupyter Notebook process(es), waiting up to TIMEOUT_SECONDS
before forcing termination.
EOF
}

PATTERN="jupyter-notebook|jupyter notebook"
TIMEOUT_SECONDS="10"
POSITIONAL_COUNT=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    -h|--help)
      usage
      exit 0
      ;;
    -t|--timeout|--TIMEOUT)
      if [[ $# -lt 2 ]]; then
        echo "[stop_notebook] Missing value for $1." >&2
        usage >&2
        exit 1
      fi
      TIMEOUT_SECONDS="$2"
      shift 2
      ;;
    --timeout=*|--TIMEOUT=*)
      TIMEOUT_SECONDS="${1#*=}"
      shift
      ;;
    --)
      shift
      while [[ $# -gt 0 ]]; do
        if [[ ${POSITIONAL_COUNT} -eq 0 ]]; then
          TIMEOUT_SECONDS="$1"
        else
          echo "[stop_notebook] Too many arguments." >&2
          usage >&2
          exit 1
        fi
        POSITIONAL_COUNT=$((POSITIONAL_COUNT + 1))
        shift
      done
      ;;
    -*)
      echo "[stop_notebook] Unknown option: $1" >&2
      usage >&2
      exit 1
      ;;
    *)
      if [[ ${POSITIONAL_COUNT} -eq 0 ]]; then
        TIMEOUT_SECONDS="$1"
      else
        echo "[stop_notebook] Too many arguments." >&2
        usage >&2
        exit 1
      fi
      POSITIONAL_COUNT=$((POSITIONAL_COUNT + 1))
      shift
      ;;
  esac
done

if ! [[ "${TIMEOUT_SECONDS}" =~ ^[0-9]+$ ]]; then
  echo "[stop_notebook] TIMEOUT_SECONDS must be a number. Got: ${TIMEOUT_SECONDS}" >&2
  exit 1
fi

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
