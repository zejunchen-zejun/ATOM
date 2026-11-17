#!/bin/bash
# make_pd_launcher.sh <NIC> <ATOM PD commit>: inside the k3pd container (prep_container.sh runs it). The image kit's
# launcher (/opt/k3/k3-mlperf-repro/make_launcher.sh: checks the image's /app/ATOM and /app/aiter commits and the
# launcher md5, writes /root/.cache/k3_server_dspark.sh); the PD ATOM tree /root/.cache/atom-pd at <commit>
# (/app/ATOM plus /kit/atom-k3pd-pd.bundle, so every tray serves the same commit); /root/.cache/k3pd_server.sh, that
# launcher on the PD tree with the K3_KV_TRANSFER_CONFIG knob (make_pd_launcher.py). Safe to rerun.
set -euo pipefail
NIC=$1 COMMIT=$2
mkdir -p /root/.cache/k3repro /root/.cache/logs /root/.cache/k3pd/steplog
cp -f /opt/k3/k3-mlperf-repro/add_*_knob.py /root/.cache/k3repro/   # 1116 adds add_prefill_chunk_knob.py
bash /opt/k3/k3-mlperf-repro/make_launcher.sh "$NIC"
D=/root/.cache/atom-pd
g() { git -c safe.directory='*' -C "$D" "$@"; }
if [ "$(g rev-parse HEAD 2> /dev/null)" != "$COMMIT" ]; then
  rm -rf "$D"
  git -c safe.directory='*' clone -q --no-checkout /app/ATOM "$D"
  g fetch -q /kit/atom-k3pd-pd.bundle 'refs/heads/*:refs/remotes/k3pd/*'
  g -c advice.detachedHead=false checkout -q --detach "$COMMIT"
fi
[ "$(g rev-parse HEAD)" = "$COMMIT" ] || { echo "$D is not at $COMMIT" >&2; exit 2; }
[ -z "$(g status --porcelain --untracked-files=no)" ] || { echo "$D has local changes" >&2; exit 2; }
python3 /kit/make_pd_launcher.py /root/.cache/k3_server_dspark.sh /root/.cache/k3pd_server.sh "$COMMIT"
# The FABRIC backend's copy kernel is JIT-compiled on first use; build it now rather than in the first transfer
cd /tmp && python3 -c 'from mori.io.fabric_copy_jit import ensure_fabric_copy_kernel as k; print("fabric copy kernel:", k())'
echo "$(hostname -s): PD tree $(g log -1 --format='%h %s' | cut -c1-70)"
