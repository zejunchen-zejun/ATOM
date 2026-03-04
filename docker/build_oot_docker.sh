#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

DOCKERFILE_PATH="${SCRIPT_DIR}/Dockerfile_For_OOT"
VLLM_REPO="${VLLM_REPO:-https://github.com/vllm-project/vllm.git}"
VLLM_DOCKER_REF="${VLLM_DOCKER_REF:-main}"
VLLM_COMMIT="f5d17400303149bbb480f6abfb6f7bb646c1d895"
IMAGE_TAG="${IMAGE_TAG:-atom-vllm-oot}"
VLLM_BASE_IMAGE="${VLLM_BASE_IMAGE:-rocm/vllm-dev:base}"
BUILD_VLLM_BASE="${BUILD_VLLM_BASE:-1}"

if [[ "${BUILD_VLLM_BASE}" == "1" ]]; then
  VLLM_TMP_DIR="$(mktemp -d)"
  trap 'rm -rf "${VLLM_TMP_DIR}"' EXIT

  echo "Step 1/2: clone vLLM and build ROCm base image"
  echo "  vLLM repo   : ${VLLM_REPO}"
  echo "  vLLM ref    : ${VLLM_DOCKER_REF}"
  echo "  Image tag   : ${VLLM_BASE_IMAGE}"

  git clone "${VLLM_REPO}" "${VLLM_TMP_DIR}"
  git -C "${VLLM_TMP_DIR}" checkout "${VLLM_DOCKER_REF}"

  VLLM_BASE_DOCKERFILE="${VLLM_TMP_DIR}/docker/Dockerfile.rocm_base"
  if [[ ! -f "${VLLM_BASE_DOCKERFILE}" ]]; then
    echo "ERROR: cannot find ${VLLM_BASE_DOCKERFILE}"
    exit 1
  fi

  DOCKER_BUILDKIT=1 docker build \
    -f "${VLLM_BASE_DOCKERFILE}" \
    -t "${VLLM_BASE_IMAGE}" \
    "${VLLM_TMP_DIR}"
fi

echo "Step 2/2: build ATOM OOT image"
echo "  Dockerfile : ${DOCKERFILE_PATH}"
echo "  Image tag  : ${IMAGE_TAG}"
echo "  Base image : ${VLLM_BASE_IMAGE}"
echo "  VLLM commit: ${VLLM_COMMIT}"

DOCKER_BUILDKIT=1 docker build \
  -f "${DOCKERFILE_PATH}" \
  -t "${IMAGE_TAG}" \
  --build-arg "BASE_IMAGE=${VLLM_BASE_IMAGE}" \
  --build-arg "VLLM_COMMIT=${VLLM_COMMIT}" \
  "$@" \
  "${REPO_ROOT}"

