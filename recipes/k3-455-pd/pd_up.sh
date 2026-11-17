#!/bin/bash
# pd_up.sh <tag>: on the router tray, start the 1P1D servers on the four serving trays: P (TP8 + DCP8, max-num-seqs
# $P_SEQS) on P_NODE0 + P_NODE1, D (TP8 + DCP8, max-num-seqs $D_SEQS) on D_NODE0 + D_NODE1. Refuses busy trays.
# Server logs: $WORK/cache/logs/k3pd_<tag>_<P|D>.log on each tray; step logs (py/sitecustomize.py, engine core only):
# $WORK/cache/k3pd/steplog/<tag>_<P|D> on each node0.
# Both instances: the image kit's recipe for the points without DSpark (agentx: prefix cache, 16384 batched tokens,
# prefill token graphs to 16384, FP8 KV with block 128, FP8 prefill attention, MoE TP8, HERD, mono off, custom
# all-reduce up to 8 MiB). ReplaySSM off: moriio moves no ReplaySSM buffers. Every tray starts with a dropped page
# cache and no loader prefetch: R1's P trays (2 TB, page cache still full of the downloaded checkpoint, prefetch on)
# went down while loading, the README's "page cache full at start" failure; D's (dropped, no prefetch) went into the
# same KFD userptr storm after loading. So the checkpoint is also read without mmap (ATOM_DISABLE_MMAP: no mapped file
# pages in the server), and memguard.sh keeps free memory up and kills a tray's server in a storm.
set -euo pipefail
cd "$(dirname "$0")" && . ./config.env
TAG=${1:?usage: pd_up.sh <tag>}
on() { ssh -o BatchMode=yes -o ConnectTimeout=10 -o LogLevel=ERROR "$1" "$2"; }
# DSPARK=k (environment): DSpark with k draft tokens on BOTH instances. ATOM's P/D design: the producer loads the
# drafter only to write the draft layers' KV (5 MLA layers in the target's pool, moved with the rest) and never
# speculates; the consumer speculates. P and D must agree on k: without ReplaySSM (moriio cannot move its records) the
# KDA state is 1 + k slots a request and the conv state k columns wider, on both ends of the transfer.
# BLOCK_SIZE (environment, default 128 as the image's start_point.sh): KV block size on BOTH instances (moriio moves
# blocks, so P and D must agree). Under DCP8 the prefix cache indexes block_size x 8 tokens at a time, and each turn
# recomputes the previous prompt's last partial unit on P (28% of P's prefill at 128, R13; p_reuse.py --hb).
ENV="-e K3_MODE=agentx -e K3_DSPARK=${DSPARK:-0} -e K3_BLOCK_SIZE=${BLOCK_SIZE:-128} -e K3_DRAFT=/draft"
ENV="$ENV -e ATOM_USE_ASM_FP8_PREFILL_ATTN=1"
ENV="$ENV -e ATOM_TOKEN_GRAPH_SIZES=256,512,1024,1536,2048,3072,4096,5120,6144,7168,8192,10240,12288,14336,16384"
ENV="$ENV -e AITER_CUSTOM_AR_MAX_SIZE=8388608 -e ATOM_FP4_MXSCALE_USE_32X4_SCALE_SHUFFLE=1"
ENV="$ENV -e RCCL_DDA_LL_ONESHOT_THRESHOLD=2097152 -e RCCL_DDA_LL_TWOSHOT_THRESHOLD=134217728"
ENV="$ENV -e NCCL_GRAPH_REGISTER=0 -e AITER_FLYDSL_USE_HERD=1 -e ATOM_MONO_ENABLE=0 -e ATOM_ENABLE_REPLAYSSM=0"
ENV="$ENV -e PYTHONPATH=/kit/py -e ATOM_LOADER_PREFETCH=false -e ATOM_DISABLE_MMAP=true"
# 1116 image (its kit's start_point.sh, every point): Lamport one-shot all-reduce up to 1 MiB, Lamport two-shot to
# 8 MiB, INT4-coded Lamport two-shot (lossy, as the MI355 recipe's quick-reduce) 8-256 MiB; DCP prefill chunk 131072.
# Not its prefill delayer (ATOM_PREFILL_DECODE_INTERVAL / _DELAYER_MAX_QUEUE_MS, c16 up): it paces prefills between the
# decodes of one instance, and here P only prefills and D only decodes.
ENV="$ENV -e AITER_CUSTOM_AR_LAMPORT_RMS_MAX_SIZE=1048576 -e AITER_CUSTOM_AR_LAMPORT_AR_MAX_SIZE=1048576"
ENV="$ENV -e AITER_CUSTOM_AR_Q4_2S_MIN_SIZE=8388608 -e AITER_CUSTOM_AR_Q4_2S_MAX_SIZE=268435456 -e K3_PREFILL_CHUNK=131072"
# P_BATCHED_TOKENS / D_BATCHED_TOKENS (environment, default 16384): max-num-batched-tokens per instance; prefill token
# graphs stop at 16384, so a larger P budget runs its biggest steps eager.
# P_MOE / D_MOE (environment, tp | ep, default tp): MoE TP8 or EP8 per instance; EP keeps attention TP8 + DCP8, so the
# P->D transfer is unchanged. P_EXTRA / D_EXTRA ("-e VAR=value ..."): other per-role additions, recorded in pd_up.txt;
# not for variables set here (a repeated -e is resolved by whoever reads the environment).
ENV_P="$ENV -e K3_MOE=${P_MOE:-tp} -e K3_MAX_BATCHED_TOKENS=${P_BATCHED_TOKENS:-16384} -e K3_MAX_SEQS=$P_SEQS -e K3_STEPLOG=/root/.cache/k3pd/steplog/${TAG}_P ${P_EXTRA:-}"
ENV_D="$ENV -e K3_MOE=${D_MOE:-tp} -e K3_MAX_BATCHED_TOKENS=${D_BATCHED_TOKENS:-16384} -e K3_MAX_SEQS=$D_SEQS -e K3_STEPLOG=/root/.cache/k3pd/steplog/${TAG}_D ${D_EXTRA:-}"
./preflight.sh || { echo "preflight failed; not starting" >&2; exit 1; }
for t in $P_NODE0 $P_NODE1 $D_NODE0 $D_NODE1; do
  on "$t" 'sync; echo 1 | sudo -n tee /proc/sys/vm/drop_caches > /dev/null'
  on "$t" "setsid -f $KIT/memguard.sh start $TAG > /dev/null 2>&1 < /dev/null"
done
printf 'pd_up %s\n  P: %s\n  D: %s\n' "$TAG" "$ENV_P" "$ENV_D"
on "$D_NODE0" "$KIT/up.sh D 0 k3pd_${TAG}_D.log $ENV_D"
on "$D_NODE1" "$KIT/up.sh D 4 k3pd_${TAG}_D.log $ENV_D"
on "$P_NODE0" "$KIT/up.sh P 0 k3pd_${TAG}_P.log $ENV_P"
on "$P_NODE1" "$KIT/up.sh P 4 k3pd_${TAG}_P.log $ENV_P"
