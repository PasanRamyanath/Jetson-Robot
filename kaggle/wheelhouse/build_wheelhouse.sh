#!/usr/bin/env bash
# Build the offline wheelhouse Kaggle dataset "<user>/beni-wheelhouse" (§10.2): cuts ~6 min of pip per session.
#   kaggle/wheelhouse/build_wheelhouse.sh [--push]
# Needs linux x86_64 + python3.12 (or docker, which is used automatically). The Kaggle CLI must be configured for --push.
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$HERE/../.." && pwd)"
OUT="${OUT:-$ROOT/build/wheelhouse}"
KUSER="${KAGGLE_USERNAME:-$(python3 -c 'import json,os;print(json.load(open(os.path.expanduser("~/.kaggle/kaggle.json")))["username"])' 2>/dev/null || echo YOUR_KAGGLE_USERNAME)}"
TS_VER="${TS_VER:-1.84.3}"
mkdir -p "$OUT/brain" "$OUT/vllm"

build() {   # SRC=repo root, OUT=wheelhouse dir; must run on linux x86_64 with python3.12
  python3 -m pip install -q -U pip uv
  cp "$(command -v uv)" "$OUT/uv"
  python3 -m pip wheel -q --no-deps -w "$OUT/brain" "$SRC/shared" "$SRC/kaggle"
  python3 -m pip download -q -d "$OUT/brain" -r "$SRC/kaggle/wheelhouse/requirements-brain.txt"
  cp "$SRC/kaggle/wheelhouse/requirements-brain.txt" "$OUT/brain/"     # the kernel installs from this list
  python3 -m pip download -q -d "$OUT/vllm" -r "$SRC/kaggle/wheelhouse/requirements-vllm.txt"
  for d in "$OUT/brain" "$OUT/vllm"; do     # sdists -> wheels so the kernel never compiles anything
    for s in "$d"/*.tar.gz; do [ -e "$s" ] && python3 -m pip wheel -q --no-deps -w "$d" "$s" && rm "$s"; done
  done
}

if [ "$(uname -s)-$(uname -m)" = "Linux-x86_64" ] && python3 -c 'import sys;assert sys.version_info[:2]==(3,12)' 2>/dev/null; then
  SRC="$ROOT" build
else
  docker run --rm -v "$ROOT:/src:ro" -v "$OUT:/out" -e SRC=/src -e OUT=/out python:3.12-slim \
    bash -c "$(declare -f build); build"
fi

curl -fsSL -o "$OUT/tailscale_${TS_VER}_amd64.tgz" "https://pkgs.tailscale.com/stable/tailscale_${TS_VER}_amd64.tgz"
curl -fsSL -o "$OUT/cloudflared" https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64
chmod +x "$OUT/cloudflared"
du -sh "$OUT"

cat > "$OUT/dataset-metadata.json" <<JSON
{"title": "beni-wheelhouse", "id": "$KUSER/beni-wheelhouse", "licenses": [{"name": "other"}]}
JSON
if [ "${1:-}" = "--push" ]; then
  if kaggle datasets status "$KUSER/beni-wheelhouse" >/dev/null 2>&1; then
    kaggle datasets version -p "$OUT" -m "wheelhouse $(date +%F)" --dir-mode zip
  else
    kaggle datasets create -p "$OUT" --dir-mode zip
  fi
fi
