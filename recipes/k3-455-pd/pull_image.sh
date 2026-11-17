#!/bin/bash
# pull_image.sh <image>: docker login with a throwaway DOCKER_CONFIG (token via stdin), pull, logout, delete the config.
set -uo pipefail
IMG=$1
export DOCKER_CONFIG=$(mktemp -d)
trap 'docker logout > /dev/null 2>&1; rm -rf "$DOCKER_CONFIG"' EXIT
printf '%s' "$K3_DOCKER_TOKEN" | docker login -u valarlip --password-stdin || exit 1
for i in 1 2 3; do docker pull "$IMG" && break; echo "pull attempt $i failed" >&2; sleep 10; done
docker image inspect "$IMG" --format '{{.Id}} {{index .RepoDigests 0}} {{.Size}}'
