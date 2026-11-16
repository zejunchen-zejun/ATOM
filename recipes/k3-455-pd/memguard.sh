#!/bin/bash
# memguard.sh start <tag> | stop: on a serving tray, around a run. R1 lost both P trays, and its D trays were heading the
# same way, to a KFD userptr storm under host-memory pressure: once the page cache had eaten the free memory, reclaim
# and compaction kept invalidating the ATOM processes' userptr mappings, KFD kept evicting their queues
# (dmesg: amdgpu_amdkfd_restore_userptr_worker hogged CPU), and the tray stopped answering.
# Every 5 s: MemFree below MEMGUARD_MIN_FREE_GB (default 64) -> drop the clean page cache (the 1.45 TB checkpoint read
# through 0.5-2 TB of RAM is what fills it). Every 30 s, one line in /data/k3pd/memguard/<tag>.log: free,
# available and cache GB, the summed KFD eviction time of this tray's ATOM processes (evicted_ms, as the emulation's
# evict_sampler.sh) and its growth, drops so far. Growth above MEMGUARD_STORM_MS (default 30000 ms per 30 s over the
# tray's processes) in two samples running: a storm; down.sh kills this tray's server to save the tray.
set -uo pipefail
cd "$(dirname "$0")" && . ./config.env
mkdir -p /data/k3pd/memguard
DIR=/data/k3pd/memguard   # the host user's: $WORK/cache/logs belongs to the container's root
PIDF=$DIR/memguard.pid
case ${1:-} in
  stop)
    [ -f "$PIDF" ] && kill "$(cat "$PIDF")" 2> /dev/null
    rm -f "$PIDF"
    exit 0
    ;;
  start) TAG=${2:?usage: memguard.sh start <tag> | stop} ;;
  *) echo "usage: memguard.sh start <tag> | stop" >&2; exit 2 ;;
esac
[ -f "$PIDF" ] && kill "$(cat "$PIDF")" 2> /dev/null
MIN=${MEMGUARD_MIN_FREE_GB:-64} STORM=${MEMGUARD_STORM_MS:-30000}
LOG=$DIR/$TAG.log
guard() {
  local i=0 drops=0 prev=-1 hot=0 ms free avail cache d
  while true; do
    free=$(awk '/^MemFree:/ {print int($2 / 1048576)}' /proc/meminfo)
    if [ "$free" -lt "$MIN" ]; then
      sync
      echo 1 | sudo -n tee /proc/sys/vm/drop_caches > /dev/null && drops=$((drops + 1))
    fi
    if [ $((i % 6)) = 0 ]; then
      ms=0
      for d in /sys/class/kfd/kfd/proc/*; do
        grep -qa "ATOM::\|openai_server" "/proc/$(basename "$d")/cmdline" 2> /dev/null || continue
        ms=$((ms + $(cat "$d"/stats_*/evicted_ms 2> /dev/null | awk '{s += $1} END {print s + 0}')))
      done
      avail=$(awk '/^MemAvailable:/ {print int($2 / 1048576)}' /proc/meminfo)
      cache=$(awk '/^Cached:/ {print int($2 / 1048576)}' /proc/meminfo)
      d=$([ "$prev" -lt 0 ] && echo 0 || echo $((ms - prev)))
      [ "$d" -lt 0 ] && d=0
      prev=$ms
      echo "$(date '+%T') free $free avail $avail cache $cache GB | evicted_ms $ms (+$d) | drops $drops" >> "$LOG"
      if [ "$d" -gt "$STORM" ]; then hot=$((hot + 1)); else hot=0; fi
      if [ "$hot" -ge 2 ]; then
        echo "$(date '+%T') STORM: evictions +$d ms in 30 s twice running; killing this tray's server" >> "$LOG"
        ./down.sh >> "$LOG" 2>&1
        hot=0
      fi
    fi
    i=$((i + 1))
    sleep 5
  done
}
guard < /dev/null > /dev/null 2>&1 &
echo $! > "$PIDF"
disown
echo "$(hostname -s): memguard on (min free $MIN GB, storm > $STORM ms / 30 s), log $LOG"
