#!/usr/bin/env bash
# MediaMTX (§5.3): re-serves the vision teleop stream (MPEG-TS on udp://127.0.0.1:5000) as WebRTC/WHEP on :8889
# for the tailnet, and as RTSP on 127.0.0.1:8554 for beni_face's picture-in-picture. Static Go binary, ~20 MB.
set -euo pipefail
VER=${MEDIAMTX_VERSION:-v1.9.3}
DEST=/usr/local/bin/mediamtx
if ! [ -x "$DEST" ] || ! "$DEST" --version 2>/dev/null | grep -q "$VER"; then
  tmp=$(mktemp -d)
  curl -fsSL "https://github.com/bluenviron/mediamtx/releases/download/${VER}/mediamtx_${VER}_linux_arm64v8.tar.gz" \
    | tar -xz -C "$tmp" mediamtx
  sudo install -m 755 "$tmp/mediamtx" "$DEST"
  rm -rf "$tmp"
fi
sudo install -d /etc/mediamtx
sudo install -m 644 "$(dirname "$0")/../systemd/mediamtx.yml" /etc/mediamtx/mediamtx.yml
sudo install -m 644 "$(dirname "$0")/../systemd/mediamtx.service" /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now mediamtx
echo "teleop: http://$(tailscale ip -4 2>/dev/null || hostname -I | cut -d' ' -f1):8889/teleop"
