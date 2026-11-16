#!/bin/bash
# router.sh up | down [file for its log] | status: atomesh, the PD router, on the router tray, in a container of the
# image (its /usr/local/bin/atomesh). ATOM PD protocol (--backend atom): the request goes to P with max_tokens 1 and
# stream off, then to D with P's kv_transfer_params. P is chosen by X-Session-ID (dp_sticky: a session stays on the P
# that holds its prefix cache; new sessions go to the P with the fewest), D round robin. One P and one D here; the same
# line takes repeated --prefill / --decode for xPyD. Request timeout 2 h: a c1024 turn can wait minutes for P or D.
set -euo pipefail
cd "$(dirname "$0")" && . ./config.env
URL=http://$ROUTER_IP:$ROUTER_PORT
case ${1:-} in
  up)
    docker rm -f "$ROUTER_CONTAINER" > /dev/null 2>&1 || true
    docker run -d --name "$ROUTER_CONTAINER" --network=host --ulimit nofile=1048576:1048576 \
      --entrypoint atomesh "$IMAGE" launch --host 0.0.0.0 --port "$ROUTER_PORT" --backend atom --pd-disaggregation \
      --prefill "http://$P_NODE0_IP:8000" --decode "http://$D_NODE0_IP:8000" \
      --prefill-policy dp_sticky --decode-policy round_robin \
      --request-timeout-secs 7200 --disable-circuit-breaker --log-level info --prometheus-port 29100 > /dev/null
    for _ in $(seq 1 60); do
      curl -sf -m 5 "$URL/health" > /dev/null && { echo "router up: $URL"; exit 0; }
      sleep 5
    done
    echo "router not answering on $URL/health:" >&2
    docker logs --tail 20 "$ROUTER_CONTAINER" >&2
    exit 1
    ;;
  down)
    if docker inspect "$ROUTER_CONTAINER" > /dev/null 2>&1; then
      [ -z "${2:-}" ] || docker logs "$ROUTER_CONTAINER" > "$2" 2>&1
      docker rm -f "$ROUTER_CONTAINER" > /dev/null
      echo "router down${2:+, log in $2}"
    fi
    ;;
  status)
    docker ps --filter "name=^$ROUTER_CONTAINER\$" --format '{{.Names}} {{.Status}}'
    curl -s -m 5 "$URL/health"; echo
    ;;
  *) echo "usage: router.sh up | down [log file] | status" >&2; exit 2 ;;
esac
