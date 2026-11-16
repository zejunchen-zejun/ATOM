#!/bin/bash
# down.sh: stop this tray's server processes, and only them. They run as root, and with --pid=host a pkill inside the
# container would reach every process on the host, so signal the PIDs docker top lists for this container. SIGTERM
# first, SIGKILL after DOWN_GRACE_S (default 45) seconds: R2 died twice on compile caches that an earlier kill -9 (and a
# kernel panic) had left half written.
cd "$(dirname "$0")" && . ./config.env
pids() { docker top "$CONTAINER" -eo pid,args 2> /dev/null | awk '/openai_server|ATOM::/ {print $1}' | paste -sd' '; }
P=$(pids)
if [ -n "$P" ]; then
  docker exec "$CONTAINER" kill -TERM $P 2> /dev/null
  for _ in $(seq 1 "${DOWN_GRACE_S:-45}"); do
    [ -z "$(pids)" ] && break
    sleep 1
  done
  left=$(pids)
  [ -z "$left" ] || docker exec "$CONTAINER" kill -9 $left
fi
sleep 3
echo "$(hostname -s): stopped [${P:-none}]${left:+ (SIGKILL: $left)}, GPU processes now $(ls /sys/class/kfd/kfd/proc 2> /dev/null | wc -l)"
