#!/bin/bash
# preflight.sh: on the router tray, check the four serving trays before a start; pd_up.sh runs it first. Per tray:
# reachable; amdgpu loaded with 4 GPUs and none in use; all 4 GPUs ACTIVE in vPOD 1 of the same PPOD; numa_balancing 0
# (set to 0 here when a reboot turned it back on, as the image kit's README asks); the k3pd container running (started
# here if a reboot stopped it) with the PD tree at ATOM_PD_COMMIT. Exits 1 if anything would stop the start.
set -uo pipefail
cd "$(dirname "$0")" && . ./config.env
on() { ssh -o BatchMode=yes -o ConnectTimeout=10 -o LogLevel=ERROR "$1" "$2"; }
CHECK='
k=$(ls -d /sys/class/kfd/kfd/topology/nodes/*/ 2> /dev/null | while read d; do grep -q "gfx_target_version 12" $d/properties && echo x; done | wc -l)
procs=$(ls /sys/class/kfd/kfd/proc 2> /dev/null | wc -l)
fab=$(for g in 0 1 2 3; do amd-smi fabric -g $g 2> /dev/null | awk -F": " "/VPOD_ID/{v=\$2} /ACCEL_STATE/{s=\$2} END{print v\"/\"s}"; done | sort -u | paste -sd,)
nb=$(cat /proc/sys/kernel/numa_balancing)
if [ "$nb" != 0 ]; then echo 0 | sudo -n tee /proc/sys/kernel/numa_balancing > /dev/null && nb="0 (was $nb)"; fi
st=$(docker inspect -f "{{.State.Running}}" '"$CONTAINER"' 2> /dev/null)
if [ "$st" = false ]; then docker start '"$CONTAINER"' > /dev/null && st="true (started)"; fi
atom=$(docker exec '"$CONTAINER"' git -c safe.directory=* -C /root/.cache/atom-pd rev-parse HEAD 2> /dev/null)
echo "gpus=$k procs=$procs fabric=$fab numa_balancing=$nb container=${st:-missing} atom=${atom:-none}"
'
ok=1
for t in $P_NODE0 $P_NODE1 $D_NODE0 $D_NODE1; do
  r=$(on "$t" "$CHECK" 2> /dev/null) || r="unreachable"
  bad=""
  case $r in
    unreachable) bad="unreachable" ;;
    *)
      [[ $r == *"gpus=4 "* ]] || bad="$bad no-4-gpus"
      [[ $r == *"procs=0 "* ]] || bad="$bad gpu-busy"
      [[ $r == *"fabric=1/ACTIVE "* ]] || bad="$bad not-vpod1-active"
      [[ $r == *"container=true"* ]] || bad="$bad no-container"
      [[ $r == *"atom=$ATOM_PD_COMMIT"* ]] || bad="$bad atom-tree"
      ;;
  esac
  echo "$t: $r${bad:+  <- FAIL:$bad}"
  [ -z "$bad" ] || ok=0
done
[ $ok = 1 ] && echo "PREFLIGHT PASS" || { echo "PREFLIGHT FAIL"; exit 1; }
