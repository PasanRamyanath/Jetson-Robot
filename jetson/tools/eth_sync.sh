#!/usr/bin/env bash
# Bench bulk sync over Gigabit Ethernet (§4 item 59): recordings, demos and memory backups -> NAS/PC, and optionally
# ONNX/models <- PC (then `make engines`: TRT engines are built on the Nano). Refuses to run over Wi-Fi, so a
# multi-GB copy never starves the voice link.
#   bash jetson/tools/eth_sync.sh you@nas:/tank/beni [--pull you@pc:~/beni-models] [--dry-run]
set -euo pipefail
DEST=${1:?usage: eth_sync.sh user@host:/path [--pull user@host:/path] [--dry-run]}; shift
PULL=""; DRY=()
while [ $# -gt 0 ]; do
  case "$1" in
    --pull) PULL=${2:?}; shift 2 ;;
    --dry-run) DRY=(--dry-run); shift ;;
    *) echo "unknown option $1" >&2; exit 2 ;;
  esac
done
IF=${BENI_ETH_IF:-eth0}
if [ "$(cat "/sys/class/net/$IF/carrier" 2>/dev/null)" != "1" ]; then
  echo "$IF has no link: plug in the bench cable (set BENI_ETH_IF for another interface)" >&2; exit 1
fi
HOST=${DEST%%:*}; HOST=${HOST#*@}
if ! ip route get "$(getent ahostsv4 "$HOST" | awk 'NR==1{print $1}')" 2>/dev/null | grep -q "dev $IF"; then
  echo "$HOST is not routed via $IF (it would go over Wi-Fi); use its LAN address" >&2; exit 1
fi
RS=(rsync -a --partial --info=progress2 "${DRY[@]}")
SRC=${BENI_DATA:-/ssd/beni}
for d in rec lerobot/raw backup logs; do
  [ -d "$SRC/$d" ] && "${RS[@]}" "$SRC/$d/" "$DEST/${d//\//_}/"   # rsync makes only the last path level
done
if [ -n "$PULL" ]; then
  "${RS[@]}" "$PULL/" "${BENI_MODELS:-$SRC/models}/"
fi
echo "eth_sync done"
