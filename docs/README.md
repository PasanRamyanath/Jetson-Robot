# Beni run guides

Step-by-step commands for getting Beni from a bare Jetson Nano and a Kaggle account to a running robot. The
blueprint ([Beni_Robot_Engineering_Blueprint.md](../Beni_Robot_Engineering_Blueprint.md)) explains *why*; these
guides say *what to type*. Do them in this order:

| # | Guide | Where | Time |
|---|---|---|---|
| 1 | [Dev PC: tests, lint, local brain](01_DEV_PC.md) | any PC (Python ≥ 3.10) | 10 min |
| 2 | [Jetson Nano setup](02_JETSON_SETUP.md) | Nano (JetPack 4.6.6 / L4T 32.7.6) | 3–4 h, mostly waiting |
| 3 | [Models and TensorRT engines (YOLO26n / YOLOv8n)](03_MODELS_ENGINES.md) | PC or Kaggle, then the Nano | 1–2 h |
| 4 | [Kaggle brain](04_KAGGLE_BRAIN.md) | Linux PC + kaggle.com | 1 h |
| 5 | [ESP32 base firmware](05_FIRMWARE.md) | PC with USB | 30 min |
| 6 | [Daily operation, tools and troubleshooting](06_OPERATION.md) | Nano | — |

Then run the phase gates in [ACCEPTANCE.md](../ACCEPTANCE.md).

## Conventions in these guides
- `nano$` is a shell on the Jetson (as your normal user with sudo), `pc$` is your dev PC, `kaggle>` is the Kaggle
  web UI.
- The repo lives at `~/beni` on the Nano; `sudo make jetson-install` links `/opt/beni` to it. All services run from
  `/opt/beni`, as user `beni`.
- Data lives on the SSD: `/ssd/beni/{models,engines,rec,logs,memory.db}`, `/ssd/maps`, `/ssd/face`.
- Secrets live in `/etc/beni/beni.env` on the Nano (mode 0600) and in Kaggle Secrets on the brain. Never commit
  them.

## Where the code differs from the blueprint text (on purpose)

| Blueprint | Code | Why |
|---|---|---|
| `configs/vision.yaml` `detector.model` (§6.4.3) | `BENI_DETECTOR` in `/etc/beni/beni.env` | One switch read by both vision services; no YAML parser on the py3.6 side |
| `configs/vision_pipeline.txt` (§5.4) | `build_pipeline()` in `jetson/vision/beni_vision.py` | The pipeline depends on flags (cams, rec, teleop, detector) |
| `configs/tracker_config.yml` (§5.4.2) | DeepStream's stock `config_tracker_NvDCF_perf.yml` | Same values; nothing to override |
| `jetson/agent/bg/scheduler.py` (§3.17) | `jetson/agent/beni_agent/scheduler/` (`beni-sched`) | One package with the agent |
| LangGraph `graph.py` (§10.7) | plain asyncio in `kaggle/beni_brain/session.py` | Lower latency and fewer dependencies (CLAUDE.md) |
| Face PiP as a DMA-BUF fd over `SCM_RIGHTS` (§4 item 29) | `beni_face` pulls vision_core's teleop H.264 from MediaMTX (RTSP loopback → NVDEC → plane 2), only while shown | GStreamer 1.14's `nvoverlaysink` path can't import a foreign NvBuffer fd; the stream is already encoded, so it costs no CPU copies and vision_core still owns all pixels |
| Offline ASR `streaming-zipformer-en-2023-06-26`, TTS Piper `amy-medium`, speaker ERes2Net (§8.3) | Moonshine tiny int8, Piper `amy-low`, 3D-Speaker CAM++ (`jetson/setup/07_models.sh`) | Offline ASR only runs on whole utterances after VAD, where Moonshine is ~3× cheaper on an A57; `low` VITS keeps TTS RTF well under 1 next to vision; CAM++ is smaller and faster at similar EER |
| One YOLO26n ONNX (§5.4.1 / §6.4.2) | Two: `yolo26n_512x288_b2.onnx` (vision_core) and `yolo26n_ds_512x288_b2.onnx` (DeepStream) | DeepStream-Yolo's parser needs its own output layout |
| Brain embeddings `bge-m3` (§10.2) | `bge-small-en-v1.5` ONNX on both sides, plus `bge-reranker-v2-m3` over the top-24 (`BENI_RERANKER`) | §11.1 requires the same embedding model on the Jetson and the brain; the reranker restores bge-m3-level precision |
| SAM2.1 masks in `vision_tools.py` (§10.2 / §10.7) | Florence-2 boxes only (open-vocabulary detection, then phrase grounding) | Pointing and `look_at` need only a box; the VLM tells similar objects apart ("which cup?") from the snapshot. That saves ~0.2 GB of GPU1 and a dependency |
| Mic-mute switch on an ESP32 GPIO (§12.3) | Optional switch on a Jetson header pin (`BENI_GPIO_MUTE`, sysfs poll in `gpio.py`) | The ESP32's free pins went to the bumpers (firmware `config.h`); the agent reacts directly, without a UART round trip |
| Hour → day → week summaries (§11.9 step 2) | Day and week summaries only (`consolidate.summarise`) | Episodes are already one summary per conversation turn; an hour layer would add LLM calls without compressing much |
| Importance re-scoring (§11.9 step 6, optional) | Not done | The extractor scores each turn with the turn's context; reflections (step 3) add the cross-episode view |
| Rule facts compiled into code constraints (§11.11) | Rules always reach the prompt (`Retriever.context`); the LLM honours them | A place keepout needs the room's polygon, which places don't have yet (§11.10, Phase 5) |
| Repo layout (§17.1): `shared/schemas.py`, `shared/memory/`, `jetson/agent/memory/db.py`, `kaggle/brain_notebook.ipynb`, `systemd/*.d/affinity.conf` | `shared/beni_common/` package (schemas, proto, hlc, memory/), one shared memory store for both sides; `kaggle/brain_notebook.py` pushed as a script kernel; `CPUAffinity=` inside each unit | Both sides import one pip-installable package instead of copies; a script kernel needs no jupytext step; one file per unit is easier to audit |
