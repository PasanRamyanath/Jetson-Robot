# 6. Daily operation, tools and troubleshooting

## 6.1 Services

| Unit | Runs | Enabled at boot |
|---|---|---|
| `beni-audio` | ALSA/AHUB routing + audio I/O (`jetson/audio/audio_io.sh`) | yes |
| `beni-vision` | DeepStream bring-up path (`beni_vision.py`, host py3.6) | yes, until Phase 3 |
| `beni-vision-core` | C++ MMAPI/TensorRT vision (Phase 3; conflicts with `beni-vision`) | after the switch ([03 §3.6](03_MODELS_ENGINES.md#36-switch-deepstream--vision_core-phase-3)) |
| `beni-face` | the LCD face (`beni_face`, no X) | yes |
| `beni-ros` | ROS 2 Humble container: bridge, SLAM, Nav2 | yes |
| `beni-agent` | voice FSM, cloud link, memory, behaviour, lifecycle (py3.8 venv) | yes |
| `beni-sched` | power modes, fan, background GPU admission, nightly jobs | yes |
| `beni-llm` | offline llama.cpp server | no, started on demand |
| `beni-vault` | unlocks the LUKS vault at boot | after `make vault` |
| `mediamtx` | WebRTC teleop on :8889, RTSP for the face PiP | yes |

```bash
nano$ cd ~/beni && make status                         # one line per unit
nano$ journalctl -u beni-agent -f                      # any unit; -b for this boot, --since "10 min ago"
nano$ sudo systemctl restart beni-agent
nano$ sudo systemctl restart beni-audio beni-vision beni-face beni-ros beni-agent beni-sched   # everything
```

After changing `/etc/beni/beni.env`, restart the units that read it: `beni-agent`, `beni-sched` and the vision
unit.

## 6.2 Updating the code

```bash
nano$ cd ~/beni && git pull
nano$ sudo make jetson-install                          # only if units, sudoers or tmpfiles changed; idempotent
nano$ .venv38/bin/pip install -e shared[memory] -e jetson/agent       # only if requirements changed
nano$ make face        # if jetson/face changed
nano$ make vision-core # if jetson/vision_core changed
nano$ sudo systemctl restart beni-agent beni-sched
```

When `shared/` or `kaggle/beni_brain/` changes, also rebuild the wheelhouse ([04 §4.4](04_KAGGLE_BRAIN.md#44-offline-wheelhouse-required)).
The next brain session picks it up.

## 6.3 Choosing the detector

```bash
nano$ sudoedit /etc/beni/beni.env          # BENI_DETECTOR=yolo26n | yolov8n | yolo26n_416 | yolov8n_416 (last two: vision_core)
nano$ sudo systemctl restart beni-vision-core    # or beni-vision
nano$ journalctl -u beni-vision-core -b | grep "\[infer\] detector"
```

If the chosen model's files are missing, both paths fall back to YOLOv8n and log it. For the bake-off, see
[03 §3.5](03_MODELS_ENGINES.md#35-bake-off-yolo26n-or-yolov8n-643).

## 6.4 Watching the cameras

Open `http://<jetson-tailscale-ip>:8889/teleop` in a browser on the tailnet. It shows both cameras side by side
(H.264, WebRTC). A new viewer triggers an IDR, so the picture appears at once. `beni-sched` sets the stream's
bitrate from the Wi-Fi signal: 2 Mbps above −62 dBm, down to 0.4 Mbps below −78 dBm, and the full rate on Ethernet.
In idle-watch and sleep replay (§15.1) vision_core stops CAM0's Argus session; opening the teleop page, a snapshot
or a thumbnail restarts it within ~1.5 s.

## 6.5 Teleop and demos

The tools talk to the services over the IPC sockets in `/tmp/beni`, which belong to `beni`. Run them as `beni`:

```bash
nano$ cd /opt/beni
nano$ sudo -u beni python3 jetson/tools/teleop.py                     # w/s a/d or arrows, q/e arcs, space stop, +/- speed
nano$ sudo -u beni python3 jetson/tools/teleop.py --joy /dev/input/js0 # gamepad (add beni to "input" if needed)
nano$ sudo -u beni python3 jetson/tools/teleop.py --record "dock at the charger"   # r start/end a demo, x discard
```

Demos are JSONL files in `/ssd/beni/lerobot/raw`. The frames come from vision_core's recordings, so keep
`beni-vision-core` recording while you collect. `beni-vision` writes `cam{i}_NNNNN.mkv`, which the exporter can't
match.

To turn demos into a LeRobot v2.1 dataset, run this on a PC or Kaggle (numpy, pyarrow, ffmpeg):

```bash
pc$ rsync -a you@beni-jetson:/ssd/beni/lerobot/raw/ raw/ && rsync -a you@beni-jetson:/ssd/beni/rec/ rec/
pc$ python jetson/tools/lerobot_export.py raw rec beni_demos        # appends new episodes; safe to re-run
```

## 6.6 Tools

| Tool | Command (Nano, from `/opt/beni`) | What for |
|---|---|---|
| Benchmarks (§14.3) | `sudo -u beni make bench` (`MODE=idle` for an idle run) | TRT engines, cameras, llama.cpp, retrieval, per-service CPU/RSS, 10 min tegrastats → `/ssd/beni/logs/bench-<date>.md` |
| tegrastats CSV | `python3 jetson/tools/tegrastats_logger.py /ssd/beni/logs/tegra.csv 500` | GPU/CPU/EMC/thermal over time (Ctrl-C to stop) |
| Voice latency | `python3 jetson/tools/voice_latency_report.py /ssd/beni/logs/turns.jsonl` | p50/p90 per stage (Phase 2 gate) |
| Base console | `python3 jetson/tools/base_console.py monitor` | ESP32 frames ([05](05_FIRMWARE.md)); stop `beni-ros` first |
| ISP overrides | `sudo python3 jetson/tools/isp_override.py show / set KEY VALUE / unset KEY / restore` | last resort for lighting problems; one key at a time, each write keeps a backup |
| Engine bench | `sudo -u beni bash jetson/engines/build_all.sh --bench` | mean / p99 per engine |
| Bench sync (§4 #59) | `bash jetson/tools/eth_sync.sh you@nas:/tank/beni [--pull you@pc:~/onnx] [--dry-run]` | recordings, demos, backups and logs → NAS over the Ethernet cable only; `--pull` fetches ONNX files, then `make engines` |
| Argus debug | `sudo systemctl stop nvargus-daemon && sudo enableCamPclLogs=5 enableCamScfLogs=5 nvargus-daemon` | camera pipeline errors |

## 6.6.1 Privacy and proactive behaviour (§12.2, §12.3)

- **Mic mute:** a switch to GND on a free header pin, with `BENI_GPIO_MUTE=<sysfs gpio>` (e.g. `194` = pin 15) in
  `/etc/beni/beni.env`. Beni stops streaming audio and the face shows the mute icon.
- **Camera privacy:** say "privacy mode" / "close your eyes" (or "open your eyes" to undo), or ask the brain.
  - The agent stops `beni-vision-core` / `beni-vision` (sudoers allows exactly this) and the eyes show `closed`.
  - The mode survives an agent restart (kv `privacy`).
  - Check it with `systemctl is-active beni-vision-core beni-vision`.
- **Proactive behaviours:**
  - greet, reminders, dock on low battery, ask an unknown face its name, ask about a recurring visitor;
  - `routine_nudge`: someone usually chats with Beni at this hour-of-week;
  - `share_memory`: "a year ago today…";
  - `lost_item`: sleep replay found something you asked about;
  - `idle_wander`: nobody around for 20 min, battery ≥ 60%, then a known place, at most every 2 h.
  - Quiet hours allow only reminders and docking. "Not now" pauses everything for 30 min.

## 6.7 Brain

- **Is it up?** `curl -s https://<relay-url>/health` (shows `brain_connected: true`),
  or `sudo -u beni /opt/beni/.venv38/bin/kaggle kernels status <you>/beni-brain`.
- **Start it now:** say the wake word, which keeps it up for 30 min, or `make kaggle-push KAGGLE_KERNEL=<you>/beni-brain`.
- **Weekly usage** is kept in the memory DB (`lifecycle.used_s.<year>-W<week>`). Change the cap with
  `BENI_WEEKLY_BUDGET_H`.

## 6.8 Troubleshooting

| Symptom | Likely cause → fix |
|---|---|
| `beni-vision` fails with `No module named zmq` | host py3.6 packages missing → re-run `bash jetson/setup/05_py38_venv.sh` (installs them system-wide) |
| `[infer] no detector` / DeepStream "failed to build engine" | engines not built → [03 §3.4](03_MODELS_ENGINES.md#34-copy-and-build-nano); check `/ssd/beni/engines/*.log` |
| Engine build killed (OOM) | vision was still running → `sudo systemctl stop beni-vision beni-vision-core`, then `make engines` |
| Cameras black / `nvargus` errors | `sudo systemctl restart nvargus-daemon` then the vision unit; check ribbon cables; `bash jetson/tools/bench/cams.sh 10` |
| No sound / mic | `aplay -l` shows `tegrasndt210ref`? jetson-io i2s4 enabled ([02 §2.5](02_JETSON_SETUP.md#25-enable-the-40-pin-functions-i2s4-pwm0-spi1))? `journalctl -u beni-audio` for amixer warnings |
| Face stays blank | `beni-face` conflicts with the desktop → `sudo systemctl set-default multi-user.target`, reboot; clips in `/ssd/face`? |
| Agent says the cloud is offline | [04 §4.8](04_KAGGLE_BRAIN.md#48-troubleshooting); meanwhile `beni-llm` answers |
| `kaggle push failed` | `/home/beni/.kaggle/kaggle.json` present and mode 600, `KAGGLE_KERNEL` set |
| Base doesn't move | `beni-ros` running? `base_console.py monitor` shows flags (e-stop/cliff/watchdog)? |
| Hot / throttling | `beni-sched` drops to idle above 80 °C; check the fan (`cat /sys/devices/pwm-fan/target_pwm`) |
| Jetson switched itself off | battery < 5 % off the charger for 60 s (§15.1): `journalctl -u beni-sched -b -1 \| grep critical`; charge it |
| Out of memory over time | `sudo -u beni make bench` per-service RSS vs §7.1; `journalctl -k | grep -i oom` |
| SSD won't boot | pick `recovery` on the serial console; the SD rootfs is untouched |

The daily/weekly acceptance checks are in [ACCEPTANCE.md](../ACCEPTANCE.md).
