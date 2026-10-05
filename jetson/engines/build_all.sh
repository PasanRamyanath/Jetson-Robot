#!/usr/bin/env bash
# Lean TensorRT 8.2 engines on the Nano (§6.2): no cuDNN/cuBLAS tactics => ~200-400 MB less RSS per process.
# Run with the camera pipeline stopped (the builder needs RAM). Skips engines that already exist; FORCE=1 rebuilds.
# Usage: build_all.sh [--bench]
set -euo pipefail
M=${BENI_MODELS:-/ssd/beni/models}; E=${BENI_ENGINES:-/ssd/beni/engines}
TRT=/usr/src/tensorrt/bin/trtexec
LEAN="--tacticSources=-CUDNN,-CUBLAS,-CUBLAS_LT"
mkdir -p "$E"
for u in beni-vision beni-vision-core; do sudo -n systemctl stop $u 2>/dev/null || true; done

build() {  # onnx engine workspace_mb
  local onnx="$M/$1" eng="$E/$2"
  [ -f "$onnx" ] || { echo "skip $1 (missing)"; return 0; }
  if [ -f "$eng" ] && [ "${FORCE:-0}" != 1 ]; then echo "have $2"; return 0; fi
  echo "== $2"
  "$TRT" --onnx="$onnx" --saveEngine="$eng.tmp" --fp16 --workspace="$3" $LEAN > "$E/${2%.engine}.log" 2>&1 \
    && mv "$eng.tmp" "$eng" || { echo "FAILED $2, see ${2%.engine}.log"; rm -f "$eng.tmp"; }
  grep -q "sufficient workspace" "$E/${2%.engine}.log" && echo "  note: raise workspace for $2" || true
}

build yolo26n_512x288_b2.onnx     yolo26n_512x288_b2_fp16.engine 1024   # vision_core primary (§6.2)
build yolo26n_ds_512x288_b2.onnx  yolo26n_ds_512x288_b2_fp16.engine 1024   # DeepStream primary (§5.4.1 export)
build yolov8n_512x288_b2.onnx     yolov8n_512x288_b2_fp16.engine 1024   # fallback (vision_core + DeepStream)
build yolo26n_416x224_b2.onnx     yolo26n_416_b2_fp16.engine    1024   # light mode (§6.1), --det yolo26n_416
build yolov8n_416x224_b2.onnx     yolov8n_416_b2_fp16.engine    1024   # light mode fallback, --det yolov8n_416
build yolov8n_face_160_b8.onnx    yolov8n_face_b8_fp16.engine   512
build scrfd_500m_320_b4.onnx      scrfd_b4_fp16.engine          512
build mobilefacenet_112_b8.onnx   mobilefacenet_b8_fp16.engine  256
build osnet_x0_25_256x128_b4.onnx osnet_b4_fp16.engine          256
build trtpose_r18_224_b4.onnx     trtpose_r18_b4_fp16.engine    256   # gestures, on demand (§4 item 39)

if [ "${1:-}" = "--bench" ]; then
  for eng in "$E"/*.engine; do
    echo "== bench $(basename "$eng")"
    "$TRT" --loadEngine="$eng" --iterations=200 --avgRuns=50 --useSpinWait --useCudaGraph 2>&1 \
      | grep -E "mean:|percentile" | head -3 || true     # pipefail: SIGPIPE / no match must not stop the loop
  done
fi
