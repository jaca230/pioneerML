#!/usr/bin/env bash
set -euo pipefail

IMAGE_NAME="pioneerml"
VERSION="${PIONEERML_VERSION:-0.1.0}"

while [[ $# -gt 0 ]]; do
  case "$1" in
    -t|--tag)
      IMAGE_NAME="$2"
      shift 2
      ;;
    -v|--version)
      VERSION="$2"
      shift 2
      ;;
    -h|--help)
      echo "Usage: build.sh [-t|--tag name] [-v|--version version]"
      exit 0
      ;;
    *)
      echo "[build.sh] Unknown option: $1"
      exit 1
      ;;
  esac
done

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

echo "[build.sh] Building ${IMAGE_NAME} (PIONEERML_VERSION=${VERSION}) from ${ROOT_DIR}"
docker build -t "${IMAGE_NAME}" --build-arg "PIONEERML_VERSION=${VERSION}" "${ROOT_DIR}"
