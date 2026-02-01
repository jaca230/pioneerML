#!/usr/bin/env bash
set -euo pipefail

IMAGE_NAME="pioneerml"
CONTAINER_NAME="pioneerml_static"
PORTS=()
STATIC=false
GPU_ARGS=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    -t|--tag)
      IMAGE_NAME="$2"
      shift 2
      ;;
    --static)
      STATIC=true
      shift
      ;;
    -p|--port)
      PORTS+=("$2")
      shift 2
      ;;
    --gpus)
      if [[ $# -lt 2 ]]; then
        echo "[run.sh] --gpus requires a value (e.g. all or 0)"
        exit 1
      fi
      GPU_ARGS=(--gpus "$2")
      shift 2
      ;;
    --gpu)
      GPU_ARGS=(--gpus all)
      shift
      ;;
    -h|--help)
      echo "Usage: run.sh [--static] [-t|--tag name] [-p|--port host:container] [--gpus all|0] [--gpu]"
      exit 0
      ;;
    *)
      echo "[run.sh] Unknown option: $1"
      exit 1
      ;;
  esac
done

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

PORT_ARGS=()
for p in "${PORTS[@]}"; do
  PORT_ARGS+=("-p" "$p")
done

if [[ "${STATIC}" == "true" ]]; then
  if docker ps -a --format '{{.Names}}' | grep -qx "${CONTAINER_NAME}"; then
    echo "[run.sh] Reusing existing container: ${CONTAINER_NAME}"
    docker start "${CONTAINER_NAME}" >/dev/null
  else
    echo "[run.sh] Creating container: ${CONTAINER_NAME}"
    docker run -dit \
      --name "${CONTAINER_NAME}" \
      -v "${ROOT_DIR}:/workspace" \
      -w /workspace \
      "${GPU_ARGS[@]}" \
      "${PORT_ARGS[@]}" \
      "${IMAGE_NAME}" >/dev/null
  fi
  docker exec -it "${CONTAINER_NAME}" /bin/bash
else
  docker run -it --rm \
    -v "${ROOT_DIR}:/workspace" \
    -w /workspace \
    "${GPU_ARGS[@]}" \
    "${PORT_ARGS[@]}" \
    "${IMAGE_NAME}"
fi
