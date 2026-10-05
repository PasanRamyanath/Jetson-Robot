#!/usr/bin/env bash
# DeepStream-Yolo parser for beni-vision (§5.4.1): clone at a pinned commit and build nvdsinfer_custom_impl_Yolo
# against DeepStream 6.0.1 / CUDA 10.2. pgie_yolo26n.txt and pgie_yolov8n.txt load it from /opt/beni/third_party.
# Pin a commit that has both DS 6.0.1 support and the YOLO26 parser: DSY_REF=<sha> bash 10_deepstream_yolo.sh.
# If the YOLO26 parser fails to compile, set BENI_DETECTOR=yolov8n for beni-vision (vision_core needs no parser).
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/../.." && pwd)
DST=${DSY_DIR:-$ROOT/third_party/DeepStream-Yolo}
REF=${DSY_REF:-}
if [ ! -d "$DST/.git" ]; then
  mkdir -p "$(dirname "$DST")"
  git clone https://github.com/marcoslucianops/DeepStream-Yolo "$DST"
fi
cd "$DST"
if [ -n "$REF" ]; then
  git fetch -q origin && git checkout -q "$REF"
elif [ ! -f "$ROOT/third_party/DeepStream-Yolo.sha" ]; then
  echo "note: no DSY_REF given; building HEAD. Pin it: DSY_REF=$(git rev-parse --short HEAD) next time."
else
  git checkout -q "$(cat "$ROOT/third_party/DeepStream-Yolo.sha")"
fi
git rev-parse HEAD > "$ROOT/third_party/DeepStream-Yolo.sha"
ls docs | grep -qi yolo26 || echo "WARNING: this commit has no YOLO26 docs; use BENI_DETECTOR=yolov8n for beni-vision"
CUDA_VER=10.2 make -C nvdsinfer_custom_impl_Yolo -j2
ls -l nvdsinfer_custom_impl_Yolo/libnvdsinfer_custom_impl_Yolo.so
echo "pinned $(cat "$ROOT/third_party/DeepStream-Yolo.sha")"
