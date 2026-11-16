#!/bin/bash
# pd_down.sh: on the router tray, stop the router and the servers on all four serving trays (down.sh on each).
cd "$(dirname "$0")" && . ./config.env
./router.sh down
for t in $P_NODE0 $P_NODE1 $D_NODE0 $D_NODE1; do
  ssh -o BatchMode=yes -o ConnectTimeout=10 -o LogLevel=ERROR "$t" "$KIT/down.sh; $KIT/memguard.sh stop"
done
