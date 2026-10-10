#!/usr/bin/env bash
# Join the tailnet as tag:beni-jetson (optional, for remote SSH / teleop from PC).
# Usage: TS_AUTHKEY=tskey-... ./04_tailscale.sh   (skipped if TS_AUTHKEY is empty)
set -euo pipefail
if [ -z "${TS_AUTHKEY:-}" ]; then
  echo "TS_AUTHKEY not set: skipping Tailscale (robot connects to cloud brain via Cloudflare Worker relay)."
  exit 0
fi
command -v tailscale >/dev/null || curl -fsSL https://tailscale.com/install.sh | sh
if tailscale status >/dev/null 2>&1; then echo "already on the tailnet"; tailscale ip -4; exit 0; fi
sudo tailscale up --authkey="$TS_AUTHKEY" --hostname=beni-jetson \
  --advertise-tags=tag:beni-jetson --accept-dns=true
tailscale ip -4
