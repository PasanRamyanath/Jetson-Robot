# TensorRT engines (§6)

Every engine is built **on the Nano** with TensorRT 8.2.1 (engines are not portable), lean
(`--tacticSources=-CUDNN,-CUBLAS,-CUBLAS_LT`, ~200-400 MB less RSS), FP16, fixed batch. The ONNX files are exported
on a PC or Kaggle (Ultralytics needs Python ≥ 3.8; the Nano has 3.6) and copied to `/ssd/beni/models`.

## Detector: YOLO26n primary, YOLOv8n fallback (§6.4)

The blueprint's detector moved from YOLOv8n to **YOLO26n** (40.9 vs 37.3 mAP, ~38 % fewer FLOPs, NMS-free,
no DFL). YOLOv8n stays as the fallback, because YOLO26 targets TensorRT 10 and may not parse or run well on 8.2.

| File in `/ssd/beni/models` | Made by | Engine | Used by |
|---|---|---|---|
| `yolo26n_512x288_b2.onnx` | `export_yolo26n.py` (Ultralytics, end-to-end `[2,300,6]`) | `yolo26n_512x288_b2_fp16.engine` | vision_core (primary) |
| `yolo26n_ds_512x288_b2.onnx` | `export_deepstream_yolo.sh` (DeepStream-Yolo `export_yolo26.py`) | `yolo26n_ds_512x288_b2_fp16.engine` | beni-vision `pgie_yolo26n.txt` |
| `yolov8n_512x288_b2.onnx` | `export_deepstream_yolo.sh` (DeepStream-Yolo `export_yoloV8.py`) | `yolov8n_512x288_b2_fp16.engine` | fallback in both paths |
| `yolo26n_416x224_b2.onnx` | `export_yolo26n.py yolo26n.pt 224 416` | `yolo26n_416_b2_fp16.engine` | vision_core light mode (§6.1): `BENI_DETECTOR=yolo26n_416` |
| `yolov8n_416x224_b2.onnx` | `SIZE="224 416" export_deepstream_yolo.sh` | `yolov8n_416_b2_fp16.engine` | its fallback (`yolov8n_416`) |

The 416×224 engines are for when the measured 512×288 b2 time blows the GPU budget (§6.1 rule). vision_core sizes
its DNN stream from the `--det` name, so the 416 fallback is used automatically with a 416 primary.

Why two YOLO26n files: DeepStream-Yolo's parser expects its own output layout, while vision_core decodes the
Ultralytics end-to-end graph directly and needs no parser. vision_core also reads DeepStream-Yolo's v8n layout
(`[B,N,6]` rows, it runs NMS itself) and EfficientNMS_TRT outputs, so one v8n file serves both paths.

Selecting the detector: `BENI_DETECTOR=yolo26n|yolov8n` in `/etc/beni/beni.env` is used by both vision services
(vision_core also takes `yolo26n_416|yolov8n_416`; beni-vision only the 512×288 pair). If the chosen model's files
are missing, vision_core falls back to the v8n engine and beni-vision to `pgie_yolov8n.txt`. This replaces the
blueprint's `configs/vision.yaml detector.model` key.

## Steps

```bash
# 1. PC or Kaggle (Python >= 3.8)
pip install -U ultralytics onnx onnxslim onnxsim
python jetson/engines/export_yolo26n.py                     # -> yolo26n_512x288_b2.onnx (checks opset and [2,300,6])
DSY_REF=$(ssh beni@beni-jetson cat /opt/beni/third_party/DeepStream-Yolo.sha) \
  bash jetson/engines/export_deepstream_yolo.sh             # -> yolo26n_ds_512x288_b2.onnx, yolov8n_512x288_b2.onnx
python jetson/engines/export_trtpose.py                     # optional: gestures (§4 item 39)
scp *.onnx labels.txt beni@beni-jetson:/ssd/beni/models/

# 2. Nano (vision is stopped for the build; the builder needs the RAM)
make engines                        # skips engines that exist; FORCE=1 rebuilds
bash jetson/engines/build_all.sh --bench                    # mean / p99 GPU compute per engine
ls /ssd/beni/engines/*.log          # "FAILED" or "raise workspace" notes are here
```

If `yolo26n` fails to parse: re-export with `OPSET=13` (never above 13), then see the §6.4.3 table
(TopK/GatherElements, C2PSA attention time, FP16 drift). If it still fails, set `BENI_DETECTOR=yolov8n`.

## Bake-off (§6.4.3, one evening)

1. Build both engines (above), then per-layer profile:
   `trtexec --loadEngine=/ssd/beni/engines/<engine> --useCudaGraph --dumpProfile --separateProfileRun`.
   If attention is > 25 % of the YOLO26n total, v8n will probably win.
2. Run each detector for a day (`BENI_DETECTOR=...`, `sudo systemctl restart beni-vision-core`) and let sleep
   replay go over the same recordings, or compare person/pet/chair recall at score 0.35 on 500 hand-checked frames.
3. **Pick YOLO26n if** it parses, p99 ≤ v8n p99 + 10 %, and recall ≥ v8n. Record the result below and in §14.3.

| Date | Engine | mean ms | p99 ms | RSS MB | recall (person/pet/chair) | Chosen |
|---|---|---|---|---|---|---|
| | yolo26n_512x288_b2_fp16 | | | | | |
| | yolov8n_512x288_b2_fp16 | | | | | |

## Other engines

| ONNX | Engine | Workspace | Notes |
|---|---|---|---|
| `yolov8n_face_160_b8.onnx` | `yolov8n_face_b8_fp16.engine` | 512 | DeepStream face SGIE |
| `scrfd_500m_320_b4.onnx` | `scrfd_b4_fp16.engine` | 512 | insightface *batched* export (`[B,N,C]` outputs) |
| `mobilefacenet_112_b8.onnx` | `mobilefacenet_b8_fp16.engine` | 256 | face embeddings |
| `osnet_x0_25_256x128_b4.onnx` | `osnet_b4_fp16.engine` | 256 | ReID for follow-me (§7.6) |
| `trtpose_r18_224_b4.onnx` | `trtpose_r18_b4_fp16.engine` | 256 | gestures, on demand |
