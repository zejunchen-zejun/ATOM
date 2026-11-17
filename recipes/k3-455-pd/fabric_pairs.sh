#!/bin/bash
# fabric_pairs.sh: from any host that reaches the four trays of config.env, mori FABRIC write tests between them, two
# disjoint pairs at a time: P0->P1 and D0->D1 (TP inside each instance), then P0->D0 and P1->D1 (the KV paths). One
# line per round; exits 1 at the first test that does not complete (logs in ${FAB_LOGS:-~/k3pd_fab}). An import
# between trays that UALink does not connect can oops the kernel: run it before a start, not next to a running one.
cd "$(dirname "$0")" && . ./config.env
L=${FAB_LOGS:-$HOME/k3pd_fab}
mkdir -p "$L"
host() { timeout "${3:-60}" ssh -o BatchMode=yes -o ConnectTimeout=10 -o LogLevel=ERROR "$1" "$2"; }
ARGS="--op-type write --all --enable-sess --enable-batch-transfer --transfer-batch-size 1"
ARGS="$ARGS --sweep-start-size 16777216 --sweep-max-size 268435456 --iters 16"
ft() {  # ft <initiator> <initiator IP> <target> <port> <tag>: one write test, its logs in $L/<tag>{,_t}.log
  (host "$3" "PORT=$4 $KIT/fabric_test.sh 1 $2 $ARGS" 400 > "$L/$5_t.log" 2>&1 &)
  host "$1" "PORT=$4 $KIT/fabric_test.sh 0 $2 $ARGS" 400 > "$L/$5.log" 2>&1
}
for pair in "$P_NODE0 $P_NODE0_IP $P_NODE1 29591 ${P_NODE0}_${P_NODE1}|$D_NODE0 $D_NODE0_IP $D_NODE1 29592 ${D_NODE0}_${D_NODE1}" \
            "$P_NODE0 $P_NODE0_IP $D_NODE0 29593 ${P_NODE0}_${D_NODE0}|$P_NODE1 $P_NODE1_IP $D_NODE1 29594 ${P_NODE1}_${D_NODE1}"; do
  a=${pair%%|*} b=${pair##*|}
  ft $a &
  ft $b
  wait
  res=""
  for t in $(echo "$a" | awk '{print $5}') $(echo "$b" | awk '{print $5}'); do
    bw=$(grep -E '^\|\s+268435456' "$L/$t.log" | awk -F'|' '{print $6}' | tr -d ' ')
    [ -n "$bw" ] || { echo "FAIL: fabric test $t did not complete (see $L/$t.log)"; exit 1; }
    res="$res $t $bw GB/s;"
  done
  echo "fabric:$res"
done
