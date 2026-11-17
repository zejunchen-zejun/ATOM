#!/bin/bash
# fabric_test.sh <node rank 0|1> <master IP> [benchmark args...]: one side of mori's two-node IO benchmark with the
# FABRIC backend (KV over UALink: both trays must be in the same vPOD) on GPU ${GPUS:-0}, in a throwaway container of
# the kit's image. fabric_pairs.sh runs it on both trays of a pair.
cd "$(dirname "$0")" && . ./config.env
R=$1 M=$2
shift 2
docker run --rm --name k3pd_fabrictest --device=/dev/kfd --device=/dev/dri --network=host --ipc=host \
  --group-add "$(getent group video | cut -d: -f3)" --group-add "$(getent group render | cut -d: -f3)" \
  --cap-add=SYS_PTRACE --cap-add=SYS_ADMIN --security-opt seccomp=unconfined --shm-size=16g \
  -e HIP_VISIBLE_DEVICES=${GPUS:-0} -e GLOO_SOCKET_IFNAME=$NIC -e MORI_SOCKET_IFNAME=$NIC \
  --entrypoint bash "$IMAGE" -c \
  "pip install -q --break-system-packages prettytable > /dev/null 2>&1; cd /app/mori && PYTHONPATH=/app/mori timeout 300 \
   torchrun --nnodes=2 --node_rank=$R --nproc_per_node=1 --master_addr=$M --master_port=${PORT:-29556} \
   -m tests.python.io.benchmark --backend fabric $*"
