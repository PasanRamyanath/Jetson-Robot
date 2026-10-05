# 1. Dev PC: tests, lint and a local brain

Any Linux, macOS or Windows PC with **Python ≥ 3.10**. The tests don't need a GPU, a Jetson or the network, because the
brain runs with stub models (`StubLLM`, `StubSTT`, `StubTTS`, `HashEmbedder`).

## 1.1 Get the code and the test dependencies

```bash
pc$ git clone <your-repo-url> beni && cd beni
pc$ python -m venv .venv && . .venv/bin/activate          # Windows: .venv\Scripts\activate
pc$ pip install pytest pytest-asyncio "websockets>=14" openai msgpack numpy pyflakes
```

Optional extras: `ffmpeg` on the PATH runs the LeRobot export end-to-end test, and `pyarrow` writes real parquet
files in it. Without them, those parts are skipped or stubbed.

## 1.2 Run the tests and the linter

```bash
pc$ make test        # or: python -m pytest -q tests          (~130 tests, ~20 s)
pc$ make lint        # or: python -m pyflakes shared kaggle jetson/agent jetson/tools jetson/vision jetson/engines tests
pc$ make e2e         # only the stub-brain websocket end-to-end test
```

What they cover:
- `tests/contract`: the ESP32 frame codec, msgpack schemas, the C++/Python schema mirror, HLC, memory sync, and the
  vision_core logic compiled for the host.
- `tests/jetson`: agent logic, gestures, places, replay, thumbnails, ISP overrides, teleop/LeRobot, detector selection.
- `tests/brain`: the text pipeline, skills, training gate, vision tools and the gateway end-to-end test.

Check the Python-version rules before committing code that runs on the Nano:

```bash
pc$ python - <<'EOF'
import ast, glob
for pat, ver in (("shared/beni_common/*.py", (3, 6)), ("jetson/vision/*.py", (3, 6)), ("jetson/tools/*.py", (3, 6)),
                 ("shared/beni_common/memory/*.py", (3, 8)), ("jetson/agent/beni_agent/**/*.py", (3, 8))):
    for p in glob.glob(pat, recursive=True):
        if p.endswith("lerobot_export.py"):
            continue                                   # PC/Kaggle tool (py3.8+), never on the Nano
        ast.parse(open(p, encoding="utf-8").read(), p, feature_version=ver)
print("ok")
EOF
```

## 1.3 Run a local brain

```bash
pc$ make brain-stub          # stub models on 127.0.0.1:8765, DB in /tmp/beni-brain
# Windows without make:
pc$ set PYTHONPATH=shared;kaggle && python -m beni_brain.gateway --stub --host 127.0.0.1 --port 8765
pc$ curl http://127.0.0.1:8765/health
```

To point a Jetson at it, run the brain on a PC on the same LAN or tailnet with `--host 0.0.0.0`, then set
`BENI_BRAIN_URL=ws://<pc-ip>:8765/ws` and the same `BENI_TOKEN` on both sides (an empty token turns auth off; only do
that on a trusted LAN).

## 1.4 PC-side jobs that support the robot

| Job | Command | Guide |
|---|---|---|
| Export detector ONNX (YOLO26n, YOLOv8n) | `python jetson/engines/export_yolo26n.py`, `bash jetson/engines/export_deepstream_yolo.sh` | [03](03_MODELS_ENGINES.md) |
| Render the face clips | `make face-clips` (numpy + x264 or ffmpeg) | [02](02_JETSON_SETUP.md#29-face-display) |
| Build and upload the brain wheelhouse | `make wheelhouse-push` (Linux x86_64 + py3.12, or docker) | [04](04_KAGGLE_BRAIN.md) |
| Flash the ESP32 base | `pio run -t upload` | [05](05_FIRMWARE.md) |
| Teleop demos to a LeRobot dataset | `python jetson/tools/lerobot_export.py raw rec out` | [06](06_OPERATION.md#65-teleop-and-demos) |
