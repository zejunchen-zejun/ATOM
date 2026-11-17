#!/bin/bash
# [C=1024] [NTRAJ=<trajectories, default whole passes of 613, at least one per user>] [CAP=10m] [WCAP=10m]
# run_pd.sh <tag>: one real 1P1D run of Kimi-K3 under the official MLPerf Endpoints client, on the router tray, detached:
#   setsid -f bash -c './run_pd.sh R1 >> /data/k3pd/results/run_pd.log 2>&1' > /dev/null 2>&1 < /dev/null
# Steps: YAMLs (validated); pd_up.sh (idle trays; P and D start together); P's and D's /health; the router; the image's
# gate through the router (P prefills, D decodes from the transferred KV and KDA state); a warmup with other seeds
# (issue cap WCAP); the measured run (submission seeds, issue cap CAP, NTRAJ trajectories) with its window; the
# summary at 16 GPUs; logs, step logs and metrics into $RESULTS/<tag>; everything stopped. Stops at the first failure
# and leaves the GPUs free. Progress: $RESULTS/run_pd.log.
# Bring-up switches: KEEP=1 leaves the servers (not the router) running after a failure, for debugging; SKIP_UP=1
# uses servers already started for this tag (pd_up.sh <tag>) instead of starting new ones.
set -uo pipefail
cd "$(dirname "$0")" && . ./config.env
TAG=${1:?usage: run_pd.sh <tag>}
C=${C:-1024} CAP=${CAP:-10m} WCAP=${WCAP:-10m}
N=${NTRAJ:-$(((C + 612) / 613 * 613))}
NW=$((C <= 613 ? C : N))
case $CAP in *m) CAPS=$((${CAP%m} * 60)) ;; *s) CAPS=${CAP%s} ;; *) echo "CAP like 10m or 600s" >&2; exit 2 ;; esac
BUDGET=$((CAPS + 2700))
export PATH=$HOME/.local/bin:$PATH
EP=$CLIENT/endpoints
RES=$RESULTS/$TAG
URL=http://$ROUTER_IP:$ROUTER_PORT
SLOG_P=k3pd_${TAG}_P.log SLOG_D=k3pd_${TAG}_D.log
mkdir -p "$RES/logs" "$RES/metrics"
on() { ssh -o BatchMode=yes -o ConnectTimeout=10 -o LogLevel=ERROR "$1" "$2"; }
say() { echo "[$(date '+%F %T')] $TAG: $*"; }
PAT='Traceback|GPU Hang|HW Exception|Memory access fault|proc died unexpectedly|Watchdog caught collective|Failed to initialize PyNcclCommunicator'
errors() {
  local n=0 t l x
  for t in "$P_NODE0 $SLOG_P" "$P_NODE1 $SLOG_P" "$D_NODE0 $SLOG_D" "$D_NODE1 $SLOG_D"; do
    set -- $t
    x=$(on "$1" "grep -cE '$PAT' $WORK/cache/logs/$2 2> /dev/null")
    n=$((n + ${x:-0}))
  done
  echo $n
}
alive() {
  local t x
  for t in $P_NODE0 $P_NODE1 $D_NODE0 $D_NODE1; do
    x=$(on "$t" "docker top $CONTAINER -eo pid,args 2> /dev/null | grep -cE 'openai_server|ATOM::'")
    [ "${x:-0}" -gt 0 ] || return 1
  done
}
router_alive() { [ "$(docker inspect -f '{{.State.Running}}' "$ROUTER_CONTAINER" 2> /dev/null)" = true ]; }
metrics() {  # metrics <when>: each instance's Prometheus metrics, if it serves them
  curl -s -m 10 "http://$P_NODE0_IP:8000/metrics" > "$RES/metrics/P_$1.txt" 2> /dev/null
  curl -s -m 10 "http://$D_NODE0_IP:8000/metrics" > "$RES/metrics/D_$1.txt" 2> /dev/null
}
collect() {
  local t
  for t in "$P_NODE0 $SLOG_P P0" "$P_NODE1 $SLOG_P P1" "$D_NODE0 $SLOG_D D0" "$D_NODE1 $SLOG_D D1"; do
    set -- $t
    scp -q "$1:$WORK/cache/logs/$2" "$RES/logs/$3.log" 2> /dev/null
    scp -q "$1:/data/k3pd/memguard/$TAG.log" "$RES/logs/memguard_$3.log" 2> /dev/null
  done
  scp -qr "$P_NODE0:$WORK/cache/k3pd/steplog/${TAG}_P" "$D_NODE0:$WORK/cache/k3pd/steplog/${TAG}_D" "$RES/logs/" 2> /dev/null
  return 0
}
stop_all() {
  ./router.sh down "$RES/logs/router.log"
  local t
  for t in $P_NODE0 $P_NODE1 $D_NODE0 $D_NODE1; do on "$t" "$KIT/down.sh; $KIT/memguard.sh stop"; done
}
fail() {
  say "FAIL: $*"
  collect
  if [ "${KEEP:-0}" = 1 ]; then
    ./router.sh down "$RES/logs/router.log"
    say "KEEP=1: servers left running (pd_down.sh stops them)"
  else
    stop_all
  fi
  say "STOPPED"
  exit 1
}
wait_healthy() {  # up to 45 minutes for both instances
  local i
  for i in $(seq 1 90); do
    curl -sf -m 5 "http://$P_NODE0_IP:8000/health" > /dev/null && curl -sf -m 5 "http://$D_NODE0_IP:8000/health" > /dev/null \
      && return 0
    [ "$(errors)" -gt 0 ] && return 1
    if [ "$i" -gt 2 ] && ! alive; then return 1; fi
    sleep 30
  done
  return 1
}
progress() { tr '\r' '\n' < "$1" 2> /dev/null | grep -oE '[0-9]+/[0-9]+ \[' | tail -1 | cut -d/ -f1; }
stop_client() {
  local i
  kill -INT "$1" 2> /dev/null
  for i in $(seq 1 12); do
    kill -0 "$1" 2> /dev/null || return 0
    sleep 5
  done
  kill -TERM "$1" 2> /dev/null
  sleep 10
}
run_client() {  # run_client <yaml> <client log> <budget s>
  local pid t0 last=-1 lastt p
  (ulimit -n 524288 && cd "$EP" && exec env UV_FROZEN=1 uv run inference-endpoint benchmark from-config \
    --config "$1" --mode perf) > "$2" 2>&1 &
  pid=$!
  t0=$(date +%s)
  lastt=$t0
  while kill -0 "$pid" 2> /dev/null; do
    sleep 60
    if [ "$(errors)" -gt 0 ] || ! alive || ! router_alive; then
      say "server error, server exit or router exit during $(basename "$1")"
      stop_client "$pid"
      return 1
    fi
    p=$(progress "$2")
    if [ "${p:-0}" != "$last" ]; then
      last=${p:-0}
      lastt=$(date +%s)
    fi
    if [ $(($(date +%s) - lastt)) -gt 1200 ]; then
      say "no client progress for 20 min"
      stop_client "$pid"
      return 1
    fi
    if [ $(($(date +%s) - t0)) -gt "$3" ]; then
      say "client over its $3 s budget"
      stop_client "$pid"
      return 1
    fi
  done
  wait "$pid"
}

WN=k3pd-c$C-warmup-$TAG
MN=k3pd-c$C-cap$CAP-$TAG
./mkyaml.sh "$WN" "$C" "$NW" "$WCAP" warmup "$RES" > "$RES/$WN.yaml"
./mkyaml.sh "$MN" "$C" "$N" "$CAP" submission "$RES" > "$RES/$MN.yaml"
for y in "$WN" "$MN"; do
  (cd "$EP" && UV_FROZEN=1 uv run inference-endpoint validate-yaml -c "$RES/$y.yaml" > /dev/null 2>&1) \
    || { say "invalid $RES/$y.yaml"; exit 1; }
done
say "1P1D c$C (P $P_NODE0+$P_NODE1 max-num-seqs $P_SEQS, D $D_NODE0+$D_NODE1 max-num-seqs $D_SEQS, KV over UALink, DSpark ${DSPARK:-0}, MoE P ${P_MOE:-tp} D ${D_MOE:-tp}, batched tokens P ${P_BATCHED_TOKENS:-16384} D ${D_BATCHED_TOKENS:-16384}${P_EXTRA:+, P_EXTRA $P_EXTRA}${D_EXTRA:+, D_EXTRA $D_EXTRA});" \
  "warmup $NW trajectories cap $WCAP, measured $N trajectories cap $CAP"
if [ "${SKIP_UP:-0}" != 1 ]; then
  ./pd_up.sh "$TAG" > "$RES/logs/pd_up.txt" 2>&1 || fail "pd_up.sh: $(tail -1 "$RES/logs/pd_up.txt")"
fi
wait_healthy || fail "servers did not come up ($(errors) error lines)"
say "P and D healthy"
./router.sh up > "$RES/logs/router_up.txt" 2>&1 || fail "router: $(tail -3 "$RES/logs/router_up.txt")"
gate=$(timeout 900 python3 gate.py "$URL" 2>&1 | tee "$RES/logs/gate.txt" | tail -1)
[[ "$gate" == *"GATE PASS"* ]] || fail "gate: $gate"
say "GATE PASS through the router; warmup"
metrics before_warmup
run_client "$RES/$WN.yaml" "$RES/$WN.log" 4500 || fail "warmup"
say "measured run, issue cap $CAP"
metrics before
date +%s.%N > "$RES/window"
run_client "$RES/$MN.yaml" "$RES/$MN.log" "$BUDGET" || fail "measured run"
date +%s.%N >> "$RES/window"
metrics after
GPUS=16 python3 summarize.py "$RES/$MN" | tee -a "$RESULTS/summary.txt"
clean=1
GPUS=16 python3 summarize.py --check "$RES/$MN" > /dev/null || clean=0
[ "$(errors)" -eq 0 ] || clean=0
collect
stop_all
[ $clean = 1 ] || { say "FAIL: the run did not end cleanly"; say "STOPPED"; exit 1; }
say "done"
