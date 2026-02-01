#!/usr/bin/env bash
set -euo pipefail

IMAGE_NAME="pioneerml"
VERSION=""
NO_CACHE=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    -v|--version)
      VERSION="$2"
      shift 2
      ;;
    -h|--help)
      echo "Usage: build.sh [-v|--version version] [--no-cache]"
      exit 0
      ;;
    --no-cache)
      NO_CACHE="--no-cache"
      shift
      ;;
    *)
      echo "[build.sh] Unknown option: $1"
      exit 1
      ;;
  esac
done

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
if [[ -z "${VERSION}" ]]; then
  VERSION="$(awk -F'\"' '/^version[[:space:]]*=/ {print $2; exit}' pyproject.toml)"
  if [[ -z "${VERSION}" ]]; then
    VERSION="0.0.0"
  fi
fi

echo "[build.sh] Building ${IMAGE_NAME} (PIONEERML_VERSION=${VERSION}) from ${ROOT_DIR}"
docker build ${NO_CACHE} -t "${IMAGE_NAME}" --build-arg "PIONEERML_VERSION=${VERSION}" "${ROOT_DIR}"
