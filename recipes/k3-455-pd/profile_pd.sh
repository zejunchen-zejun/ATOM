#!/bin/bash
# profile_pd.sh <P|D> [seconds, default 20] [out dir]: on the router tray, while a run is in its steady state, record a
# torch profile of one instance (all eight ranks) for <seconds> through ATOM's /start_profile and /stop_profile on
# its node0, then copy the traces of both trays to <out dir> (default $RESULTS/profile_<P|D>_<time>) and summarize
# them with trace_top.py. The traces land in the container's /root/.cache/torch_profile (the launcher's
# --torch-profiler-dir), i.e. $WORK/cache/torch_profile on each tray.
# Kernels inside CUDA graphs show as graph launches only: for a kernel-level view of P's prefill, start that run with
# P_EXTRA="-e ATOM_TOKEN_GRAPH_SIZES=" (prefill eager; its step times then differ from the graphed ones).
set -euo pipefail
cd "$(dirname "$0")" && . ./config.env
ROLE=${1:?usage: profile_pd.sh <P|D> [seconds] [out dir]}
SECS=${2:-20}
case $ROLE in
  P) N0=$P_NODE0 N1=$P_NODE1 IP=$P_NODE0_IP ;;
  D) N0=$D_NODE0 N1=$D_NODE1 IP=$D_NODE0_IP ;;
  *) echo "usage: profile_pd.sh <P|D> [seconds] [out dir]" >&2; exit 2 ;;
esac
OUT=${3:-$RESULTS/profile_${ROLE}_$(date +%m%d_%H%M%S)}
mkdir -p "$OUT"
on() { ssh -o BatchMode=yes -o ConnectTimeout=10 -o LogLevel=ERROR "$1" "$2"; }
since=$(date +%s)
curl -sf -m 60 -X POST "http://$IP:8000/start_profile" > /dev/null || { echo "start_profile failed on $IP" >&2; exit 1; }
echo "$(date +%T) profiling $ROLE for ${SECS}s"
sleep "$SECS"
curl -sf -m 600 -X POST "http://$IP:8000/stop_profile" | head -c 400; echo
sleep 5
for t in $N0 $N1; do   # traces written since the start, from both trays (one directory per rank)
  on "$t" "cd $WORK/cache/torch_profile 2> /dev/null && find . -type f -newermt @$since \\( -name '*.json' -o -name '*.json.gz' \\)" |
    while read -r f; do
      mkdir -p "$OUT/$t/$(dirname "$f")"
      scp -q "$t:$WORK/cache/torch_profile/$f" "$OUT/$t/$f"
    done
done
echo "traces in $OUT:"; find "$OUT" -type f | head -20
docker run --rm -v "$OUT":/t:ro -v "$KIT":/kit:ro --entrypoint python3 "$IMAGE" /kit/trace_top.py /t | tee "$OUT/summary.txt"
