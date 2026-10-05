#!/usr/bin/env bash
# Join the tailnet as tag:beni-jetson (§9.1). ACL: tag:beni-jetson -> tag:beni-brain:8765 only.
# Usage: TS_AUTHKEY=tskey-... ./04_tailscale.sh   (no-op once joined)
set -euo pipefail
command -v tailscale >/dev/null || curl -fsSL https://tailscale.com/install.sh | sh
if tailscale status >/dev/null 2>&1; then echo "already on the tailnet"; tailscale ip -4; exit 0; fi
sudo tailscale up --authkey="${TS_AUTHKEY:?set TS_AUTHKEY}" --hostname=beni-jetson \
  --advertise-tags=tag:beni-jetson --accept-dns=true
tailscale ip -4
