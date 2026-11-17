#!/bin/bash
# recover.sh <tag>: run on a host that serves nothing (not one of the four trays), detached, with config.env and
# fabric_pairs.sh next to it. Brings the four serving trays of config.env up as one UALink generation and starts run
# <tag>. UALink only connects trays activated together: a tray from another generation (warm-rebooted, or never cycled
# with the rest) crashes importing from them (IMPORT: connection setup failed -> oops -> panic), and every set of trays
# power-cycled and loaded together has connected. The router tray carries the operator's session, so this runs where
# its power cycle cannot stop it.
#   1. sync; DC power-off of the four through their BMCs (ipmitool on the BMC, as a DC power-off on Helios-R); wait
#      until every BMC reports off; power-on.
#   2. The Startup steps on all four at once: amdgpu loaded in parallel (no flags: the BKC for this driver);
#      wait for 4 ACTIVE AIFM devices, the full station mask and 4 active accelerators; NUMA balancing off; k3pd up;
#      the runtime caches of the trays in CLEAN (default: none) wiped, as after a crash.
#   3. Pairwise mori FABRIC tests (fabric_pairs.sh): P0->P1 and D0->D1 (TP inside each instance), then P0->D0 and
#      P1->D1 (KV). Stops at the first failure.
#   4. All passed: KEEP=1 ./run_pd.sh <tag> on the router tray. Progress in ~/k3pd_recover.log here.
# RUN_ENV ("VAR=value ..."): the run's knobs, e.g. RUN_ENV="P_MOE=ep BLOCK_SIZE=64". RECOVER_DELAY_S (default 120): set 0
# when this host is not one of the four (a router tray that serves nothing can run this; then its session survives).
# The BMC credentials come from ~/.k3pd_secret (askpass.sh, bmc_known_hosts) and are deleted at the end.
set -u
TAG=${1:?usage: recover.sh <tag>}
cd "$(dirname "$0")" && . ./config.env
D=$HOME/.k3pd_secret LOG=$HOME/k3pd_recover.log
NODES="$P_NODE0 $P_NODE1 $D_NODE0 $D_NODE1"
say() { echo "[$(date '+%F %T')] $TAG: $*" >> "$LOG"; }
bmc() {
  SSH_ASKPASS=$D/askpass.sh SSH_ASKPASS_REQUIRE=force DISPLAY=none timeout 40 ssh \
    -o UserKnownHostsFile="$D/bmc_known_hosts" -o StrictHostKeyChecking=accept-new \
    -o PreferredAuthentications=password,keyboard-interactive -o PubkeyAuthentication=no -o ConnectTimeout=10 \
    -o LogLevel=ERROR "$BMC_USER@$(printf "$BMC_HOST_FMT" "$1")" "$2" 2> /dev/null
}
host() { timeout "${3:-60}" ssh -o BatchMode=yes -o ConnectTimeout=10 -o LogLevel=ERROR "$1" "$2"; }
finish() { rm -rf "$D"; say "$1"; exit "${2:-0}"; }

sleep "${RECOVER_DELAY_S:-120}"   # time for the operator's session on the router tray to report before it goes down
say "start: cold power cycle of $NODES (P $P_NODE0 + $P_NODE1, D $D_NODE0 + $D_NODE1)"
for n in $NODES; do host "$n" sync 30 || true; done
for n in $NODES; do say "$n power off: $(bmc "$n" 'ipmitool power off' | tail -1)"; done
for i in $(seq 1 60); do
  off=0
  for n in $NODES; do [[ "$(bmc "$n" 'ipmitool power status' | tail -1)" == *"is off"* ]] && off=$((off + 1)); done
  [ $off = 4 ] && break
  sleep 10
done
[ $off = 4 ] || finish "FAIL: only $off/4 chassis off" 1
sleep 30
for n in $NODES; do say "$n power on: $(bmc "$n" 'ipmitool power on' | tail -1)"; done
for i in $(seq 1 150); do
  up=0
  for n in $NODES; do host "$n" true 15 && up=$((up + 1)); done
  [ $up = 4 ] && break
  sleep 10
done
[ $up = 4 ] || finish "FAIL: only $up/4 hosts back" 1
say "all four booted; loading amdgpu on all four at once"
sleep 20
for n in $NODES; do (host "$n" 'sudo -n modprobe amdgpu' 180 > /dev/null 2>&1) & done
wait
CONV='a=$(sudo -n afmctl show device 2>/dev/null | grep -c "Config phase *: ACTIVE"); m=$(sudo -n cat /sys/class/drm/card*/device/ualink/station_lane_en_bitmap 2>/dev/null | sort -u | paste -sd,); s=$(sudo -n cat /sys/class/drm/card*/device/ualink/accel_state 2>/dev/null | grep -c active); echo "$a $m $s"'
for n in $NODES; do
  ok=0
  for i in $(seq 1 60); do
    r=$(host "$n" "$CONV" 30)
    [ "$r" = "4 ffffffffffffffffff 4" ] && { ok=1; break; }
    sleep 5
  done
  [ $ok = 1 ] || finish "FAIL: $n fabric not ready: [$r]" 1
  host "$n" 'echo 0 | sudo -n tee /proc/sys/kernel/numa_balancing > /dev/null; docker start k3pd > /dev/null' 60
  say "$n ready: aifm/mask/accel [$r], numa_balancing 0, k3pd up"
done
for t in ${CLEAN:-}; do host "$t" "$KIT/clean_caches.sh" 120 >> "$LOG" 2>&1; done

# Pairwise UALink tests (fabric_pairs.sh, next to this script): inside P and inside D, then the two KV paths
./fabric_pairs.sh > "$HOME/k3pd_fab_$TAG.log" 2>&1
ok=$?
while read -r l; do say "$l"; done < "$HOME/k3pd_fab_$TAG.log"
[ $ok = 0 ] || finish "FAIL: fabric tests (logs in ~/k3pd_fab)" 1
host "$ROUTER" "setsid -f bash -c 'cd $KIT && ${RUN_ENV:-} KEEP=1 ./run_pd.sh $TAG >> $RESULTS/run_pd.log 2>&1' > /dev/null 2>&1 < /dev/null"
finish "DONE: $TAG launched on $ROUTER (KEEP=1); progress in $RESULTS/run_pd.log there"
