#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage: start_lab.sh [--port PORT] [--host HOST]
       start_lab.sh [PORT] [HOST]

Start JupyterLab in the background.

Arguments:
  --port, -p  TCP port to bind (default: 8888)
  --host      Interface/IP to bind (default: 0.0.0.0)
  PORT HOST   Positional fallback for compatibility

Environment:
  JUPYTER_LOG_DIR  Directory for log files
                   (default: <repo>/.runtime/jupyter)
EOF
}

PORT="8888"
HOST="0.0.0.0"
POSITIONAL_COUNT=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    -h|--help)
      usage
      exit 0
      ;;
    -p|--port|--PORT)
      if [[ $# -lt 2 ]]; then
        echo "[start_lab] Missing value for $1." >&2
        usage >&2
        exit 1
      fi
      PORT="$2"
      shift 2
      ;;
    --port=*|--PORT=*)
      PORT="${1#*=}"
      shift
      ;;
    --host|--HOST)
      if [[ $# -lt 2 ]]; then
        echo "[start_lab] Missing value for $1." >&2
        usage >&2
        exit 1
      fi
      HOST="$2"
      shift 2
      ;;
    --host=*|--HOST=*)
      HOST="${1#*=}"
      shift
      ;;
    --)
      shift
      while [[ $# -gt 0 ]]; do
        if [[ ${POSITIONAL_COUNT} -eq 0 ]]; then
          PORT="$1"
        elif [[ ${POSITIONAL_COUNT} -eq 1 ]]; then
          HOST="$1"
        else
          echo "[start_lab] Too many arguments." >&2
          usage >&2
          exit 1
        fi
        POSITIONAL_COUNT=$((POSITIONAL_COUNT + 1))
        shift
      done
      ;;
    -*)
      echo "[start_lab] Unknown option: $1" >&2
      usage >&2
      exit 1
      ;;
    *)
      if [[ ${POSITIONAL_COUNT} -eq 0 ]]; then
        PORT="$1"
      elif [[ ${POSITIONAL_COUNT} -eq 1 ]]; then
        HOST="$1"
      else
        echo "[start_lab] Too many arguments." >&2
        usage >&2
        exit 1
      fi
      POSITIONAL_COUNT=$((POSITIONAL_COUNT + 1))
      shift
      ;;
  esac
done

if ! [[ "${PORT}" =~ ^[0-9]+$ ]]; then
  echo "[start_lab] PORT must be a number. Got: ${PORT}" >&2
  exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
DEFAULT_LOG_DIR="${REPO_ROOT}/.runtime/jupyter"
LOG_DIR="${JUPYTER_LOG_DIR:-${DEFAULT_LOG_DIR}}"
LOG_FILE="${LOG_DIR}/jupyter_lab_${PORT}.log"

mkdir -p "${LOG_DIR}"

echo "[start_lab] Starting JupyterLab on ${HOST}:${PORT}"
echo "[start_lab] Log: ${LOG_FILE}"

OLD_PIDS=$(pgrep -f "jupyter-lab|jupyter lab|jupyterlab" || true)
if [[ -n "${OLD_PIDS}" ]]; then
  echo "[start_lab] Stopping existing JupyterLab: ${OLD_PIDS}"
  kill ${OLD_PIDS} || true
fi

nohup jupyter lab \
  --ip="${HOST}" \
  --port="${PORT}" \
  --no-browser \
  --allow-root \
  --NotebookApp.allow_origin='*' \
  --ServerApp.token='' \
  --ServerApp.password='' \
  > "${LOG_FILE}" 2>&1 &

PID=$!
echo "[start_lab] PID: ${PID}"

for i in {1..10}; do
  if grep -q "http" "${LOG_FILE}"; then
    break
  fi
  sleep 1
done

if grep -a -q "http" "${LOG_FILE}"; then
  echo "[start_lab] URLs:"
  grep -a -Eo "http://[^ ]+" "${LOG_FILE}" | sed 's/\\x1b\\[[0-9;]*m//g'
else
  echo "[start_lab] Waiting for Jupyter to emit URLs..."
  tail -n 5 "${LOG_FILE}"
fi
