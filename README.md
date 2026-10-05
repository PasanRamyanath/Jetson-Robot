# Beni home robot

The code for the robot described in [Beni_Robot_Engineering_Blueprint.md](Beni_Robot_Engineering_Blueprint.md). The blueprint is the source of truth, and the `§` numbers in the code point to its sections. There are two halves:

| Half | Hardware | Runs |
|---|---|---|
| **Robot** (`jetson/`) | Jetson Nano 4 GB, JetPack 4.6.6, py3.6 host + py3.8 venv | vision, wake word, VAD, audio I/O, identity, the memory system of record, proactive behaviour, the offline brain, power modes, the Kaggle lifecycle |
| **Brain** (`kaggle/`) | Kaggle batch kernel, 2× T4, py3.12 | Qwen2.5-VL-7B-AWQ on vLLM (GPU0), faster-whisper + Kokoro/CosyVoice2 + Florence-2 (GPU1), websocket gateway, memory learning |
| **Shared** (`shared/`) | both | ESP32 frame codec, msgpack schemas, HLC, SQLite memory store + LWW sync, retrieval, embeddings |

```
robot mic ─► KWS/VAD ─► pcm16/opus ─► [tailnet ws /ws] ─► STT ─► recall ─► LLM+tools ─► sentence TTS ─► tts.chunk ─► speaker
                     ◄──────── action{move_to, look_at, …} / vision.request ◄────────┘
memory.db (Jetson, source of truth) ◄── memory.delta/ack (HLC, LWW) ──► mirror.db (brain: extraction + consolidation)
```

Step-by-step guides: [docs/](docs/README.md) (dev PC, Jetson setup, models and engines, Kaggle brain, firmware,
daily operation). Acceptance gates: [ACCEPTANCE.md](ACCEPTANCE.md).

## Layout

```
shared/beni_common/        proto.py (ESP32 COBS+CRC), schemas.py (msgpack frames), hlc.py, tegrastats.py
  memory/                  store.py + schema.sql (SQLite, FTS5, tombstones), retrieval.py (hybrid), embed.py (bge-small ONNX), sync.py (DeltaSync)
jetson/
  agent/beni_agent/        py3.8 agent: main.py, voice/ (FSM, audio, sherpa-onnx), cloud/ (link, opus, lifecycle),
                           memory/ (snapshot upload, HF backup, learned keepout), perception/ (identity, world, places),
                           behaviour/ (proactive, bandits, offline), scheduler/
  ros2_ws/src/             ROS 2 Humble (§7.4): beni_base_driver (ESP32 link + cmd_vel mux), beni_head (pan/tilt),
                           beni_zmq_bridge (robot_cmd/robot_state <-> Nav2, follow/look/search/dock), beni_description
                           (URDF), beni_bringup (one component container: SLAM, Nav2, drivers, bridge + EKF, sllidar)
  docker/                  Dockerfile.ros (Humble on r32.7.1 base, gcc-9, headless Nav2 subset) + cyclonedds.xml
  vision/                  DeepStream bring-up pipeline (py3.6) + nvinfer configs (pgie_yolo26n / pgie_yolov8n)
  vision_core/             Phase-3 C++14/CUDA vision (§5.5): Argus, VIC, TensorRT CUDA graphs, ByteTrack, SCRFD+MFN,
                           NVENC recording/teleop/motion vectors; portable logic is host-tested in tests/contract
  face/                    beni_face (C++14): expression clips via NVDEC on DRM plane 0, pupils/status icons on plane 1,
                           camera PiP on plane 2; assets/make_clips.py renders the H.264 clips on a PC
  setup/                   01 system tune · 02 root on SSD · 03 docker · 04 tailscale · 05 py3.8 venv · 06 AHUB routing ·
                           07 models · 08 MediaMTX (teleop WebRTC) · 09 vault · 10 DeepStream-Yolo parser
  engines/ llm/ audio/     TensorRT exports/builds (engines/README.md: YOLO26n vs v8n), llama.cpp, ALSA/AHUB audio
  systemd/                 units, env example, tmpfiles, sudoers
  tools/                   base console, tegrastats logger, voice latency report, bench/ (§14.3: `make bench`)
kaggle/
  beni_brain/              gateway.py (websockets), session.py (turn pipeline), stt.py, tts.py, llm.py, tools.py, vision_tools.py, prompts.py,
                           memory/ (mirror, extract, consolidate), shutdown.py
  brain_notebook.py        the Kaggle script kernel (venvs, tailscale, vLLM, gateway, keep-alive, clean exit)
  kernel-metadata.json     T4×2, internet on; the Jetson rewrites the ids from KAGGLE_KERNEL on push
  wheelhouse/              offline wheel dataset builder
firmware/base_esp32/       ESP-IDF 5.3 motor/sensor MCU (PID, encoders, bumpers, cliffs, VL53L0X, battery, watchdog)
tests/                     contract (shared), jetson (agent logic), brain (text pipeline + stub gateway e2e)
```

## Develop (any PC)

```bash
pip install pytest pytest-asyncio "websockets>=14" openai msgpack numpy pyflakes
make test          # ~130 tests, ~20 s; the brain runs with stub STT/TTS/LLM
make lint
make brain-stub    # local brain on 127.0.0.1:8765 for poking at the agent
```

## Deploy

**Brain (once, then automatic):**
1. Add Kaggle Secrets: `BENI_TOKEN`, `TS_AUTHKEY` (tagged `tag:beni-brain`), and optionally `HF_TOKEN` and `CF_TUNNEL_TOKEN`.
2. `make wheelhouse-push` on a Linux or Docker machine uploads `<you>/beni-wheelhouse` (required: the kernel installs offline from it).
3. Set `KAGGLE_KERNEL=<you>/beni-brain` in `/etc/beni/beni.env`. The Jetson's lifecycle manager pushes the kernel during awake hours or on the wake word. It asks the brain to stop (`brain.stop`) when idle or when the weekly budget runs out. The brain then consolidates memory, flushes deltas and exits.

**Robot:**
```bash
sudo make ssd-root PART=/dev/sda1 CONFIRM=1 && sudo reboot
sudo make jetson-install     # /opt/beni -> repo, user, units, env, sudoers
TS_AUTHKEY=tskey-... make jetson-setup && sudo reboot   # then jetson-io: i2s4, pwm0, spi1
make llama models engines ros-image face
sudoedit /etc/beni/beni.env && sudo systemctl start beni-agent beni-sched
```

**Board features:**
- **Fan and thermal (§15.2):** `beni-sched` drives `/sys/devices/pwm-fan`. The fan runs at low PWM up to 55 °C and
  ramps to full at 70 °C. Above 80 °C, the scheduler drops to idle mode until the SoC is back under 72 °C.
- **GPIO (§13.4):** the ESP32 e-stop line on pin 13 becomes an `estop` event. For a mic-mute button, wire it to GND
  and set `BENI_GPIO_MUTE`.
- **Touch (§4 item 57):** a USB-HID panel is found through evdev. Otherwise an XPT2046 on SPI1 (`/dev/spidev0.0`,
  enabled with jetson-io) is read directly, with no DT overlay. Tapping the face starts listening; stroking it
  "pets" Beni. For an SPI panel, set `BENI_GPIO_PENIRQ=13` (pin 22) and `BENI_TOUCH_CAL` to calibrate.
- **Phone presence (§4 item 56):** pair each phone once (`bluetoothctl`), then set `BENI_PHONES=Name=MAC,...`. When
  a phone arrives home, the brain is pre-warmed (except in quiet hours), and the prompt context gets `phones at home`.
- **Vault (§4 item 58):** `make vault` creates a LUKS2 container for `memory.db` and its backups. It holds the face
  and voice exemplars and every fact. The key lives on the microSD card, with a passphrase slot for recovery.
  `beni-vault.service` unlocks it at boot, before the agent starts.

## Spatial memory (§11.10)

- Places are rows in the Jetson DB. `save_place` ("remember this is the kitchen") stores the current SLAM pose.
  `move_to` and `goto_dock` resolve names to poses in the agent, and the bridge only ever sees x/y/yaw.
- The bridge projects detections onto the lidar scan and publishes their map positions in `robot_state.objs` at 2 Hz.
  The agent turns them into rate-limited `object_sighting` rows and per-label `object_belief` rows, which is what
  `find_object` answers from.
- Every goto/dock logs a `nav_experience` row. Each night, DBSCAN over the stuck positions, plus places of kind
  `keepout`, writes `/ssd/maps/keepout.{pgm,yaml}`. Nav2 loads it as a keepout filter on the next ROS start.

## Face (§3.11)

`beni_face` owns the HDMI LCD with no X and no GPU. NVDEC decodes the current expression clip onto plane 0. When
the expression changes, the next frame tick starts the new clip at its IDR frame. The pupils and status icons are
drawn on the CPU into a 400×240 RGBA buffer, but only when something moves. VIC then upscales it onto plane 1.

- **What sets the expression:**
  - The voice FSM drives the base expression: `listening`, `thinking`, `talking`, then `idle_blink`, or `sleepy`
    in the idle and critical power modes.
  - `set_expression` from the brain shows a clip for 4 s.
- **Gaze:** the pupils follow the largest face on cam0, and wander when nobody is there.
- **Icons:** they appear only when noteworthy: low battery or charging, weak Wi-Fi, mic muted, cloud offline.
- **Control socket:** other processes can PUSH to `face_ctrl`:
  - `{op: expression, name, hold}`
  - `{op: gaze, x, y, hold}`
  - `{op: overlay, mic_muted, cloud_offline}`
  - `{op: pip, on}`: plays MediaMTX's `teleop` stream in a 320×180 corner.

```bash
make face-clips        # dev PC: numpy + x264 -> jetson/face/assets/clips/*.h264 (+ .eyes pupil sidecars)
make face              # Nano: build, install clips to /ssd/face; then sudo systemctl restart beni-face
```

## Brain voice and vision options
These are Kaggle Secrets (a pushed kernel has no other settings), see [docs/04_KAGGLE_BRAIN.md](docs/04_KAGGLE_BRAIN.md).
- `BENI_TTS=kokoro` (default) or `cosyvoice`. CosyVoice2-0.5B speaks in a cloned voice with emotion instructions. It needs a
  Kaggle input dataset containing `beni_voice*.wav` (3-10 s of clean speech). If that file is missing or CosyVoice fails
  to load, the brain uses Kokoro for the whole session. It never mixes voices within a session.
- Florence-2-large backs the `locate` tool ("look at the red cup"). It runs grounding on a fresh snapshot and turns the
  bounding box into a relative head bearing (`look_at` with `dpan`/`dtilt`). Set `BENI_VISION=none` to turn it off.

## Benchmarks (§14.3)
`sudo -u beni make bench` on the Nano appends a markdown table to `/ssd/beni/logs/bench-<date>.md`:
TensorRT engines (mean/p99 ms, peak RSS, cuDNN/cuBLAS not mapped), dual 720p60 capture, llama.cpp tok/s on cores 2-3,
memory retrieval over 10k episodes, per-service CPU/RSS against the §7.1 budgets, and 10 min of tegrastats (mean/p95).
It stops beni-vision for the camera and TRT runs and restarts it afterwards. Each script also runs on its own.

## Style training (§11.12)
Set the Kaggle secret or env `BENI_ADAPTER_REPO` (a private HF model repo) and `BENI_LORA=1`. When the gateway stops
with at least 90 min of the session left, `beni_brain.training.run` trains on the freed GPU1. It needs at least 300 new
good turns for SFT (mixed 50/30/20 with the core persona set and older turns) or at least 100 corrections for DPO.
It loads the candidate into the running vLLM and scores it on the held-out suite (tool, persona, recall, safety, json,
language). It publishes only if no category drops below the current model. `manifest.json` in the repo records
`active`, `history` and `trained_until`, and the next session serves `active`. To undo the last adapter, run
`python -m beni_brain.training.run --repo <repo> --rollback`. Facts are never trained in; they stay in the memory DB.

## Skills (§11.9 step 11)
Every tool-using turn is logged as an `action` episode: the trigger, the tool steps and whether they succeeded.
Nightly consolidation (`beni_common.memory.skills.mine`) turns a repeated multi-step procedure into a `skill` row. The
bar is at least 3 successes with the same trigger and the same steps; one-off tools like `remember` don't count. When
the user says something close to a skill's trigger, the prompt gets a "What worked before" line. A skill that fails 3
times, and at least as often as it worked, is demoted and not proposed again.

## Teleop and demos (§11.13, LeRobot)
`jetson/tools/teleop.py` drives over SSH (w/s/a/d or arrows, `--joy /dev/input/js0` for a gamepad), sending
`{op: drive}` to the ROS bridge at 10 Hz.

With `--record "dock at the charger"`, `r` starts and ends a demo, and `x` discards it. Each demo is a small JSONL in
`/ssd/beni/lerobot/raw` with 10 Hz state and action. No frames are copied: vision_core's recordings already have them.

On a PC or Kaggle, rsync `raw/` and `/ssd/beni/rec/`, then run
`python jetson/tools/lerobot_export.py raw rec beni_demos`. It needs pyarrow and ffmpeg, and builds a LeRobot v2.1
dataset:
- per-episode parquet;
- H.264 clips of both cameras, cut from the `.ts` segments, frame-aligned to the rows;
- `meta/` with info, episodes, tasks and episode stats.

This is the input for the Diffusion Policy docking experiment. Keep recordings running while collecting demos.

## vision_core (§5.5, Phase 3)

`jetson/vision_core` is the C++ replacement for the DeepStream bring-up (`jetson/vision/beni_vision.py`). It publishes
the same `vision.sock` / `vision_ctrl.sock` messages, so the agent, the scheduler and the ROS bridge work with either.
- **Capture:** Argus with two ISP streams per camera: 1280x720 at 60 fps, and 512x288 RGBA for the detector.
- **Detection:** one CUDA-graph TensorRT launch per tick for both cameras. It uses YOLO26n (end-to-end `[2,300,6]`),
  falling back to `yolov8n_512x288_b2_fp16.engine`, then ByteTrack predicts at 60 Hz.
- **Faces:** SCRFD finds the faces, then an Umeyama warp and MobileFaceNet produce the embeddings. They are paced
  per track (5 Hz while new, then 1 Hz), and the largest face drives CAM0 auto-exposure.
- **ReID (§7.6 follow-me):** OSNet-x0.25 (`osnet_b4_fp16.engine`) embeds people on the low-priority stream. It runs
  only in active mode, and only while `beni-sched` admits background GPU work on `sched.sock`. When a track
  reappears within 30 s of ByteTrack dropping it, it keeps its old `tid`, so the target survives occlusions.
  Turn it off with `--no-reid`; it is also off when the engine is missing.
- **Gestures (§4 item 39):** TRT-Pose ResNet18 (`trtpose_r18_b4_fp16.engine`, from
  `jetson/engines/export_trtpose.py`) also runs on the low-priority stream, and only on demand. The agent asks for it
  with `{op: pose, cam, secs}` for 20 s after a wake and for 10 s when someone comes into view. Up to 4 people per
  camera are posed at 5 Hz and published on the `pose` topic. The agent detects:
  - **wave:** a raised hand swinging side to side. Beni smiles and starts listening, as with a tap on the face.
  - **point:** a straight arm held out sideways. "Nimal pointed to my left" goes into the turn context.

  Turn it off with `--no-pose`.
- **Recording and teleop:**
  - H.265 recording in 5-minute `.ts` segments under `/ssd/beni/rec`, pruned to `--rec-keep-gb`.
  - H.264 teleop (both cameras side by side) goes to MediaMTX on `udp://127.0.0.1:5000`.
  - Every tracked head gets a -6 QP ROI in the teleop encoder, so faces stay sharp at 2 Mb/s.
  - Each new viewer triggers MediaMTX `runOnRead` → `jetson/tools/vision_idr.py` → `{op: idr}`, so the picture
    appears at once instead of after up to one GOP. Both vision paths support it.
  - A 320x180 NVENC motion-vector encoder on CAM1 wakes the detector.
- **Memory thumbnails (§3.17):** `{op: thumb}` does a VIC crop and scale, then NVJPG. The agent stores a 128×128
  crop for each new face exemplar and a 320×180 keyframe on "sighting" episodes (at most one per person per
  15 min). They are saved under `thumbs/` next to `memory.db`, which is inside the vault when there is one. Orphans
  and keyframes over 256 MB are swept hourly.
- **Night:** CAM1 night logic switches to gray input and ramps the IR ring (pin 32 PWM).
- **Sleep replay (§11.9 step 9):** at night, when Beni is docked, charging and nobody has been around for 10 min,
  `beni-sched` switches to `replay` mode. The agent then feeds each finished recording segment from the last 36 h
  to `{op: replay}`. NVDEC decodes every 6th frame, and the detector and face chain run on the low-priority stream,
  pausing whenever the scheduler withdraws background GPU work. Results arrive on the `replay` topic.
  - Known people seen only in the recordings get a "sighting" episode.
  - Unknown faces are clustered across nights (kv `replay_strangers`). A stranger seen on two different days
    becomes an `unknown_visitor` episode with a keyframe, and Beni later asks "who was the person who visited
    yesterday afternoon?".
  - The keyframe goes to the brain (`replay.flag`). Florence-2 captions it and looks for objects Beni was asked to
    find and couldn't; the answer (`replay.result`) is added to the episode.
  - Recordings are read from `BENI_REC_DIR` (default `/ssd/beni/rec`). Turn replay off with `--no-replay`.

Build and switch over on the Nano:
1. Export on a PC with `python jetson/engines/export_yolo26n.py` and `bash jetson/engines/export_deepstream_yolo.sh`
   (the v8n fallback), then copy the files to `/ssd/beni/models` ([jetson/engines/README.md](jetson/engines/README.md)).
2. Run `make engines`, then `make vision-core`. `BENI_DETECTOR=yolo26n|yolov8n` in beni.env picks the detector for
   both vision services once the §6.4.3 bake-off is done.
3. Run `sudo make jetson-install`.
4. Run `sudo systemctl disable --now beni-vision && sudo systemctl enable --now beni-vision-core`.

The two units conflict, so only one of them owns the cameras. SCRFD must be the insightface *batched* export
(`[B,N,C]` outputs). Motion-vector units (`Encoder::mv_scale`) and the NoIR white balance (`CamCfg::wb`) need one
calibration pass each on the real hardware.

ISP overrides (§4 item 9) are a last resort, for when the logs show a lighting problem that `wbmode`, the AE region
and the night logic can't fix (auto-exposure hunting, crushed shadows). `jetson/tools/isp_override.py` edits
`/var/nvidia/nvcam/settings/camera_overrides.isp` one key at a time:
- `show`, `set KEY VALUE`, `unset KEY`, `restore [FILE]`;
- every write keeps a timestamped `.bak-*` and restarts `nvargus-daemon` and the vision unit;
- change one parameter, then watch `journalctl -u beni-vision-core -f` before changing the next.

To see what Argus is doing, run `sudo systemctl stop nvargus-daemon && sudo enableCamPclLogs=5 enableCamScfLogs=5
nvargus-daemon`.

## Sources

- vLLM on Kaggle T4: https://github.com/kaggle-vllm/kaggle-vllm, https://gist.github.com/Nishant0905/d1c5ff79a4622f8d0db747db3e759e84
- Qwen2.5-VL on vLLM: https://qwen.readthedocs.io/en/latest/deployment/vllm.html
- Kernel metadata: https://github.com/Kaggle/kaggle-api/blob/main/docs/kernels_metadata.md
- Kokoro TTS: https://github.com/hexgrad/kokoro
- CosyVoice2: https://github.com/FunAudioLLM/CosyVoice
- Florence-2: https://huggingface.co/microsoft/Florence-2-large
- sherpa-onnx models: https://github.com/k2-fsa/sherpa-onnx/releases
- rootOnUSB: https://github.com/jetsonhacks/rootOnUSB
