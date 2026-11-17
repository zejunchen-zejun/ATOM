#!/bin/bash
# bootstrap_node.sh <tray>: on the router tray (it holds the kit), make a tray that just joined ready to serve.
# Safe to rerun: every step checks first. The docker token comes only from the environment:
#   K3_DOCKER_TOKEN=<token> [SRC=<idle tray with the model>] ./bootstrap_node.sh <new tray>
# SRC: the tray the model is copied from (default this one). Name an idle tray when this one is busy: the data then
# flows SRC -> tray directly (SRC needs ssh access to the tray), and this tray only drives it.
#   1. access: ssh, passwordless sudo, the docker group; vPOD 1 ACTIVE (amdgpu loaded; see recover.sh); numa_balancing 0
#   2. the image ($IMAGE) through pull_image.sh (a throwaway DOCKER_CONFIG, deleted after the pull)
#   3. the model: $MODEL_DIR (Kimi-K3 1.5 TB, Kimi-K3-DSpark 6.7 GB) copied from SRC, 8 rsync streams, then
#      compared file by file by size; skipped when the tray already holds the same files
#   4. the kit, then prep_container.sh (container, PD tree at ATOM_PD_COMMIT, launcher, fabric copy kernel)
set -euo pipefail
cd "$(dirname "$0")" && . ./config.env
T=${1:?usage: bootstrap_node.sh <tray>}
ME=$(whoami)
on() { ssh -o BatchMode=yes -o ConnectTimeout=10 -o LogLevel=ERROR "$T" "$@"; }
say() { echo "[$(date +%T)] $T: $*"; }

on true || { say "no ssh access for $ME (authorized_keys needs $(awk '{print $3}' ~/.ssh/id_ed25519.pub))"; exit 2; }
on 'sudo -n true' 2> /dev/null || { say "no passwordless sudo"; exit 2; }
on 'docker ps > /dev/null' 2> /dev/null || { say "$ME is not in the docker group"; exit 2; }
fab=$(on 'for g in 0 1 2 3; do amd-smi fabric -g $g 2>/dev/null | awk -F": " "/VPOD_ID/{v=\$2} /ACCEL_STATE/{s=\$2} END{print v\"/\"s}"; done | sort -u | paste -sd,')
say "fabric ${fab:-none}"
[ "$fab" = 1/ACTIVE ] || say "WARNING: not vPOD 1 ACTIVE; load amdgpu with the other trays (recover.sh) before serving"
on 'echo 0 | sudo -n tee /proc/sys/kernel/numa_balancing > /dev/null'

if on "docker image inspect $IMAGE > /dev/null 2>&1"; then
  say "image present"
else
  : "${K3_DOCKER_TOKEN:?set K3_DOCKER_TOKEN for docker login}"
  scp -q "$KIT/pull_image.sh" "$T:/tmp/pull_image.sh"
  # The token goes over ssh's stdin, never into a command line or a file
  printf '%s\n' "$K3_DOCKER_TOKEN" |
    on 'read -r K3_DOCKER_TOKEN; export K3_DOCKER_TOKEN; bash /tmp/pull_image.sh '"$IMAGE"' | tail -1; rm -f /tmp/pull_image.sh' ||
    { say "pull failed"; exit 1; }
fi

SRC=${SRC:-}
src() { if [ -n "$SRC" ]; then ssh -o BatchMode=yes -o ConnectTimeout=10 -o LogLevel=ERROR "$SRC" "$@"; else bash -c "$*"; fi; }
LIST="find $MODEL_DIR -type f -printf '%s %P\\n' | sort -k2"
on "sudo -n mkdir -p $MODEL_DIR /data/k3pd && sudo -n chown $ME $MODEL_DIR /data/k3pd"
TH=$(ssh -G "$T" | awk '/^hostname /{print $2}')   # the name SRC can reach the tray by
if [ "$(src "$LIST")" = "$(on "$LIST")" ]; then
  say "model present"
else
  say "copying $(src "du -sh $MODEL_DIR" | cut -f1) of model from ${SRC:-this tray} with 8 streams"
  src "cd $MODEL_DIR && find . -type f -printf '%P\\n' | xargs -P 8 -I{} rsync -a --relative $MODEL_DIR/./{} $ME@$TH:$MODEL_DIR/"
  [ "$(src "$LIST")" = "$(on "$LIST")" ] || { say "model copy differs"; exit 1; }
  say "model copied"
fi

rsync -a --delete --exclude .git "$KIT/" "$T:$KIT/"
on "cd $KIT && ./prep_container.sh" | tail -3
on "cd $KIT && ./clean_caches.sh"
say "ready: image, model, kit, k3pd container at $(on "docker exec $CONTAINER git -c safe.directory='*' -C /root/.cache/atom-pd rev-parse --short HEAD")"
