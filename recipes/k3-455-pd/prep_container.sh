#!/bin/bash
# prep_container.sh: on a serving tray, create the k3pd server container from the pinned image (or check the existing
# one), then make the PD ATOM tree and launcher in it (make_pd_launcher.sh). Safe to rerun; run it on all four.
# No /dev/infiniband: the KV moves over UALink (mori FABRIC), which needs only /dev/kfd and /dev/dri.
set -euo pipefail
cd "$(dirname "$0")" && . ./config.env
docker image inspect "$IMAGE" > /dev/null 2>&1 || { echo "no image $IMAGE here: docker pull it" >&2; exit 2; }
mkdir -p "$WORK/cache" "$WORK/triton"
MOUNTS="$MODEL_DIR:/models $MODEL_DIR/Kimi-K3-DSpark:/draft $WORK/cache:/root/.cache $WORK/triton:/root/.triton $KIT:/kit"
if docker inspect "$CONTAINER" > /dev/null 2>&1; then
  have=" $(docker inspect -f '{{range .Mounts}}{{.Source}}:{{.Destination}} {{end}}' "$CONTAINER")"
  for m in $MOUNTS; do
    case $have in *" $m "*) ;; *) echo "$CONTAINER exists but does not mount $m: docker rm it" >&2; exit 2 ;; esac
  done
  [ "$(docker inspect -f '{{.Image}}' "$CONTAINER")" = "$(docker image inspect -f '{{.Id}}' "$IMAGE")" ] \
    || { echo "$CONTAINER is not from $IMAGE: docker rm it" >&2; exit 2; }
  [ "$(docker inspect -f '{{.State.Running}}' "$CONTAINER")" = true ] || docker start "$CONTAINER" > /dev/null
else
  docker run -d --name "$CONTAINER" -w /tmp \
    --device=/dev/kfd --device=/dev/dri --network=host --ipc=host --pid=host \
    --group-add "$(getent group video | cut -d: -f3)" --group-add "$(getent group render | cut -d: -f3)" \
    --cap-add=SYS_PTRACE --cap-add=SYS_ADMIN --security-opt seccomp=unconfined --shm-size=128g --ulimit memlock=-1:-1 \
    -v "$MODEL_DIR:/models:ro" -v "$MODEL_DIR/Kimi-K3-DSpark:/draft:ro" -v "$WORK/cache:/root/.cache" \
    -v "$WORK/triton:/root/.triton" -v "$KIT:/kit:ro" --entrypoint sleep "$IMAGE" infinity > /dev/null
fi
docker exec "$CONTAINER" bash /kit/make_pd_launcher.sh "$NIC" "$ATOM_PD_COMMIT"
