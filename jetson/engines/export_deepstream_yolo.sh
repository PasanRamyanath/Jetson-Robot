#!/usr/bin/env bash
# §5.4.1 DeepStream-Yolo ONNX exports for beni-vision. Run on a PC or Kaggle (Ultralytics needs Python >= 3.8),
# never on the Nano. Use the same DeepStream-Yolo commit as the Nano (third_party/DeepStream-Yolo.sha).
#   bash export_deepstream_yolo.sh [yolo26n.pt] [yolov8n.pt]      (SIZE="224 416" for the light-mode 416x224 files)
#   -> yolo26n_ds_512x288_b2.onnx   (pgie_yolo26n.txt; its parser layout, NOT vision_core's end-to-end graph)
#   -> yolov8n_512x288_b2.onnx      (pgie_yolov8n.txt, and vision_core's fallback: [B,N,6] rows + NMS)
#   -> labels.txt
# Copy them to /ssd/beni/models on the Nano, then `make engines`.
set -euo pipefail
W26=${1:-yolo26n.pt}; W8=${2:-yolov8n.pt}
DSY=${DSY_DIR:-DeepStream-Yolo}; REF=${DSY_REF:-}
OPSET=${OPSET:-12}                                   # 13 if 12 fails; never > 13 (TRT 8.2)
read -r H W <<< "${SIZE:-288 512}"
[ -d "$DSY/.git" ] || git clone https://github.com/marcoslucianops/DeepStream-Yolo "$DSY"
[ -z "$REF" ] || git -C "$DSY" checkout -q "$REF"
echo "DeepStream-Yolo $(git -C "$DSY" rev-parse --short HEAD)"
pip install -q -U ultralytics onnx onnxslim onnxsim

export_one() {  # script-glob weights dest
  local s; s=$(ls "$DSY"/utils/$1 2>/dev/null | head -1)
  [ -n "$s" ] || { echo "no $1 in this DeepStream-Yolo commit"; return 1; }
  local out; out=$(basename "${2%.pt}").onnx         # the scripts write <weights name>.onnx to the cwd
  cp "$s" . && rm -f "$out"
  # -s is H W in current commits (check `python $(basename "$s") -h` on yours)
  python3 "$(basename "$s")" -w "$2" --opset "$OPSET" -s "$H" "$W" --batch 2 --simplify
  mv "$out" "$3"
  python3 -c "import onnx,sys; m=onnx.load(sys.argv[1]); print(sys.argv[1], [o.version for o in m.opset_import], \
[[d.dim_value for d in o.type.tensor_type.shape.dim] for o in m.graph.output])" "$3"
}

export_one "export_yolo26*.py" "$W26" "yolo26n_ds_${W}x${H}_b2.onnx" || echo "YOLO26 export failed: use yolov8n in DS"
export_one "export_yoloV8.py" "$W8" "yolov8n_${W}x${H}_b2.onnx"
ls -l ./*.onnx labels.txt 2>/dev/null || true
