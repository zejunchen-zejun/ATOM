#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

DOCKERFILE_PATH="${SCRIPT_DIR}/Dockerfile_For_OOT"
VLLM_COMMIT="f5d17400303149bbb480f6abfb6f7bb646c1d895"
IMAGE_TAG="${IMAGE_TAG:-atom-vllm-oot:f5d17400}"

echo "Building OOT image with:"
echo "  Dockerfile : ${DOCKERFILE_PATH}"
echo "  Image tag  : ${IMAGE_TAG}"
echo "  VLLM commit: ${VLLM_COMMIT}"

docker build \
  -f "${DOCKERFILE_PATH}" \
  -t "${IMAGE_TAG}" \
  --build-arg "VLLM_COMMIT=${VLLM_COMMIT}" \
  "$@" \
  "${REPO_ROOT}"

