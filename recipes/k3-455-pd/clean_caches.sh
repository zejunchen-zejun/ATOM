#!/bin/bash
# clean_caches.sh: on a serving tray, wipe the server container's runtime JIT/compile caches. Run it after any crash,
# kill -9 or power loss: a kernel or graph written halfway is not always empty, and the next start loads it as is. R2
# died twice that way: P on R1's 0-byte torch.compile entries (EOFError: Ran out of input -> InductorError), D on a
# Triton kernel cut short by R1's kill -9 (HIP "device kernel image is invalid"). Kept: the PD ATOM tree, the
# launchers and mori's fabric copy kernel (built by prep_container.sh before any run). Everything wiped is rebuilt on
# the next start, which takes a few minutes longer.
set -euo pipefail
cd "$(dirname "$0")" && . ./config.env
docker exec "$CONTAINER" sh -c 'rm -rf /root/.triton/cache /root/.cache/atom /root/.cache/comgr \
  /root/.cache/huggingface/modules /tmp/torchinductor_root /tmp/aiter_configs'
echo "$(hostname -s): runtime caches wiped"
