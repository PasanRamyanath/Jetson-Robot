# 3. Models and TensorRT engines (§6)

ONNX files are exported on a **PC or Kaggle**. Ultralytics and torch need Python ≥ 3.8; the Nano has 3.6. The
engines are built **on the Nano**, because TensorRT engines aren't portable. The detector table, the reason
there are two YOLO26n files, and the bake-off record are in [jetson/engines/README.md](../jetson/engines/README.md).

## 3.1 What changed with YOLO26n

The blueprint's detector moved from YOLOv8n to **YOLO26n**: NMS-free, no DFL, `[B,300,6]` end-to-end output, and
40.9 vs 37.3 mAP at ~38 % fewer FLOPs. YOLOv8n is kept as the **fallback**, because YOLO26 targets TensorRT 10
and the Nano has 8.2.1. The project now handles this as follows:

| Piece | YOLO26n (primary) | YOLOv8n (fallback) |
|---|---|---|
| DeepStream config | `jetson/vision/configs/pgie_yolo26n.txt`: `cluster-mode=4` (no NMS) | `pgie_yolov8n.txt`: `cluster-mode=2`, `nms-iou-threshold=0.45` |
| vision_core engine | `yolo26n_512x288_b2_fp16.engine` (Ultralytics end-to-end export) | `yolov8n_512x288_b2_fp16.engine` |
| DeepStream engine | `yolo26n_ds_512x288_b2_fp16.engine` (DeepStream-Yolo export) | same v8n engine as above |
| Selection | `BENI_DETECTOR=yolo26n` (default) | `BENI_DETECTOR=yolov8n`, or automatic when the YOLO26n files are missing |
| Light mode (§6.1) | `BENI_DETECTOR=yolo26n_416` (vision_core only) | `yolov8n_416` |

## 3.2 Export the detectors (PC or Kaggle)

```bash
pc$ python -m venv ~/yolo && . ~/yolo/bin/activate
pc$ pip install -U ultralytics onnx onnxslim onnxsim
pc$ mkdir -p ~/onnx && cd ~/onnx

# vision_core YOLO26n: end-to-end [2,300,6], opset 12 (OPSET=13 at most for TRT 8.2)
pc$ python ~/beni/jetson/engines/export_yolo26n.py                  # -> yolo26n_512x288_b2.onnx
pc$ python ~/beni/jetson/engines/export_yolo26n.py yolo26n.pt 224 416   # optional light mode -> yolo26n_416x224_b2.onnx

# DeepStream YOLO26n + the YOLOv8n fallback, with the DeepStream-Yolo commit the Nano built its parser from
pc$ export DSY_REF=$(ssh you@beni-jetson cat beni/third_party/DeepStream-Yolo.sha)
pc$ bash ~/beni/jetson/engines/export_deepstream_yolo.sh            # -> yolo26n_ds_512x288_b2.onnx yolov8n_512x288_b2.onnx labels.txt
pc$ SIZE="224 416" bash ~/beni/jetson/engines/export_deepstream_yolo.sh   # optional -> yolov8n_416x224_b2.onnx
```

On Kaggle, run the same commands in a notebook cell with `!`, then download the files from `/kaggle/working`.

## 3.3 Face, ReID and pose models

These have no export script in the repo. `build_all.sh` passes no `--shapes`, so every ONNX must have a **static
batch** that matches its file name.

| File in `/ssd/beni/models` | Source | Notes |
|---|---|---|
| `scrfd_500m_320_b4.onnx` | insightface `scrfd_500m_bnkps`, exported with `detection/scrfd/tools/scrfd2onnx.py` at 320×320 | Must be the **batched** export (`[B,N,C]` outputs, 5 landmarks) |
| `mobilefacenet_112_b8.onnx` | insightface `buffalo_s`/`buffalo_sc` `w600k_mbf.onnx` (512-D) | Batch 8 |
| `osnet_x0_25_256x128_b4.onnx` | torchreid `osnet_x0_25` (ImageNet or MSMT17 weights) → `torch.onnx.export`, opset 12 | Batch 4, 256×128 |
| `yolov8n_face_160_b8.onnx` | a YOLOv8n-face export | Only for the DeepStream face SGIE |
| `trtpose_r18_224_b4.onnx` | `python jetson/engines/export_trtpose.py resnet18_baseline_att_224x224_A_epoch_249.pth` | Gestures, on demand |

To pin a dynamic batch dimension to a fixed size:

```bash
pc$ python - scrfd_500m_dyn.onnx scrfd_500m_320_b4.onnx 4 <<'EOF'
import sys, onnx
from onnxsim import simplify
src, dst, b = sys.argv[1], sys.argv[2], int(sys.argv[3])
m = onnx.load(src)
for i in m.graph.input:
    i.type.tensor_type.shape.dim[0].dim_value = b      # clears dim_param
m, ok = simplify(m)
assert ok
onnx.save(m, dst)
print([[d.dim_value for d in o.type.tensor_type.shape.dim] for o in m.graph.output])
EOF
```

Any engine whose ONNX file is missing is skipped. vision_core then turns that feature off: without the OSNet
engine there's no ReID, and without TRT-Pose there are no gestures.

## 3.4 Copy and build (Nano)

```bash
pc$   scp ~/onnx/*.onnx ~/onnx/labels.txt you@beni-jetson:/tmp/
nano$ sudo install -o beni -g beni -m 644 /tmp/*.onnx /tmp/labels.txt /ssd/beni/models/
nano$ cd ~/beni && make engines                 # as beni; stops beni-vision/-core, ~5–15 min per engine
nano$ grep -l FAILED /ssd/beni/engines/*.log; ls -la /ssd/beni/engines/*.engine
nano$ sudo -u beni bash jetson/engines/build_all.sh --bench       # mean / p99 GPU ms per engine
nano$ sudo systemctl start beni-vision           # or beni-vision-core after Phase 3
```

To rebuild after a new export: `sudo -u beni env FORCE=1 bash jetson/engines/build_all.sh`. Alternatively, delete that
one `.engine` file and run `make engines`.

**If YOLO26n fails to parse or build** (the log shows TopK, GatherElements or an unsupported op):
1. Re-export with `OPSET=13 python export_yolo26n.py` (never go above 13 for TRT 8.2), then run `FORCE=1`.
2. If it still fails, set `BENI_DETECTOR=yolov8n` in `/etc/beni/beni.env` and restart vision. Even without that,
   both services fall back to v8n on their own when the YOLO26n files are missing.

## 3.5 Bake-off: YOLO26n or YOLOv8n (§6.4.3)

Follow [jetson/engines/README.md § Bake-off](../jetson/engines/README.md#bake-off-643-one-evening):
1. Profile both engines per layer (`trtexec --dumpProfile`). If attention takes more than 25 % of the YOLO26n time,
   v8n will probably win.
2. Run each detector for a day (`BENI_DETECTOR=...`, then restart the vision unit), and compare recall on the same
   recordings.
3. Keep YOLO26n if p99 ≤ v8n p99 + 10 % and recall ≥ v8n. Record the numbers in the README table.

## 3.6 Switch DeepStream → vision_core (Phase 3)

```bash
nano$ cd ~/beni && make vision-core                    # ~15 min; needs nvidia-l4t-jetson-multimedia-api
nano$ sudo systemctl disable --now beni-vision && sudo systemctl enable --now beni-vision-core
nano$ journalctl -u beni-vision-core -f                # "[infer] detector /ssd/beni/engines/..."
```

Both vision services read `BENI_DETECTOR` from `/etc/beni/beni.env`. `beni-vision` accepts only the 512×288 pair
(`yolo26n`, `yolov8n`).
