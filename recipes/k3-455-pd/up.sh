#!/bin/bash
# up.sh <P|D> <first TP rank on this tray: 0|4> <log name> [-e VAR=value ...]: start this tray's half of the prefill
# (P) or decode (D) instance in the background, logging to $WORK/cache/logs/<log name>. Both trays of an instance get
# the same -e list (pd_up.sh starts all four). The --kv-transfer-config JSON is built here from config.env, so it never
# passes through ssh quoting: moriio over the UALink fabric (backend "fabric"), write mode (K3 keeps per-request KDA
# state, which moriio moves only by push), and on D the hosts of P's TP ranks (remote_tp_hosts: P spans two trays).
set -euo pipefail
cd "$(dirname "$0")" && . ./config.env
ROLE=$1 RANK0=$2 LOG=$3
shift 3
KV='"kv_connector": "moriio", "backend": "fabric", "moriio_mode": "write", "handshake_port": '$HANDSHAKE_PORT
case $ROLE in
  P) MASTER=$P_NODE0_IP KV="{\"kv_role\": \"kv_producer\", $KV}" ;;
  D) MASTER=$D_NODE0_IP
     KV="{\"kv_role\": \"kv_consumer\", $KV, \"remote_tp_hosts\": {\"$P_NODE0_IP\": [\"$P_NODE0_IP\", \"$P_NODE1_IP\"]}}" ;;
  *) echo "usage: up.sh <P|D> <0|4> <log name> [-e VAR=value ...]" >&2; exit 2 ;;
esac
python3 -c 'import json, sys; json.loads(sys.argv[1])' "$KV" || { echo "bad kv config: $KV" >&2; exit 2; }
[ "$(docker inspect -f '{{.State.Running}}' "$CONTAINER" 2> /dev/null)" = true ] || docker start "$CONTAINER" > /dev/null
docker exec "$CONTAINER" test -f /root/.cache/k3pd_server.sh || { echo "no launcher: run prep_container.sh" >&2; exit 2; }
docker exec -d -e "K3_KV_TRANSFER_CONFIG=$KV" "$@" "$CONTAINER" \
  bash -c "bash /root/.cache/k3pd_server.sh $RANK0 $MASTER > /root/.cache/logs/$LOG 2>&1"
echo "$(hostname -s): $ROLE TP ranks $RANK0-$((RANK0 + 3)) (node0 $MASTER), log $WORK/cache/logs/$LOG, kv $KV"
