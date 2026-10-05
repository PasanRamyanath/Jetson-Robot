# On-robot acceptance checklist (§16)

Each §16 gate has a pass condition and a way to measure it, using the tools in this repo. Run these on the Nano,
after `make jetson-setup`, `make engines` and `sudo make jetson-install`. Items marked *manual* need a person and a
tape measure or a stopwatch.

## Phase 0: Platform
- [ ] `MemAvailable` ≥ 3.3 GB at idle: `grep MemAvailable /proc/meminfo`.
- [ ] Dual 720p60 for 10 min at 59–60 fps with CPU < 15 %: `bash jetson/tools/bench/cams.sh 600` checks each sensor's
      fps (vision stopped). Check both at once with vision_core running and `tegrastats_logger.py`.
- [ ] `aplay -l` lists `tegrasndt210ref`.

## Phase 1: Body
- [ ] 2 m straight with < 5 cm error: `python3 jetson/tools/base_console.py drive 200 0 10` (*manual* tape).
- [ ] Watchdog stop < 300 ms: kill the driver mid-drive and measure the stop (*manual*, or watch `base_console.py monitor`).
- [ ] Cliff stop at a table edge, 10 out of 10 tries (*manual*).
- [ ] Teleop latency < 250 ms on the LAN: open `http://<jetson>:8889/teleop`, drive with `jetson/tools/teleop.py`, and
      film a stopwatch.
- [ ] 1 h of recording with no dropped segments: count `cam*_*.ts` in `/ssd/beni/rec` (12 per camera per hour).

## Phase 2: Voice, face and brain v1
- [ ] 20-turn conversation with p50 end-to-end < 2.0 s: `python3 jetson/tools/voice_latency_report.py
      /ssd/beni/logs/turns.jsonl`.
- [ ] Barge-in < 350 ms, and Beni doesn't wake itself at full volume in 100 tries (*manual* count).
- [ ] KWS < 0.5 false accepts per hour over 8 h (*manual*: count wakes in the agent journal with nobody talking).
- [ ] Face clip switch < 100 ms; zero audio xruns in 1 h (`journalctl -u beni-audio`).

## Phase 3: Vision core and memory v1
- [ ] `vision_core` RSS ≤ 180 MB and GPU ≤ 65 % in the foreground: `make bench`, or
      `bash jetson/tools/bench/procs.sh 30` plus `jetson/tools/tegrastats_logger.py`.
- [ ] 5 people at ≥ 95 % precision and ≥ 80 % recall, +10 % recall under window backlight compared with beni-vision
      (*manual* sessions).
- [ ] Detects a person in a dark room at 3 m with the IR ring on.
- [ ] Memory survives a brain restart; "forget me" removes the person on both sides.

## Phase 4: Space
- [ ] Maps the house; 5 named places ≥ 90 %; follows a person 20 m through 2 doors.
- [ ] Wakes from idle within 150 ms of motion (motion-vector wake).
- [ ] "Where are my keys?" is answered from sightings.

## Phase 5: Lifelong learning and full use
- [ ] Sleep replay covers a full day per night at 80–85 % GPU with no throttling: `tegrastats_logger.py` during
      `replay` mode.
- [ ] Offline brain answers within 3 s of losing Kaggle (pull the network).
- [ ] Proactive greetings > 60 % engagement after 2 weeks (bandit stats in the memory DB).
- [ ] The adapter passes the eval gate or is auto-rejected (`manifest.json` in the adapter repo).
- [ ] ISP overrides only if the logs show a lighting problem: `jetson/tools/isp_override.py`.

## Phase 6: Polish
- [ ] Auto-dock success ≥ 90 %.
- [ ] 24 h unattended with no OOM, no crash and no watchdog firing: `journalctl -k | grep -i oom`, `systemctl --failed`.
- [ ] Collect docking demos (`teleop.py --record "dock at the charger"`), then run `lerobot_export.py` on Kaggle.

## Calibration that code can't do
- Gesture thresholds (`perception/gestures.py`), the NoIR white balance (`CamCfg::wb`) and the motion-vector scale
  (`Encoder::mv_scale`).
- Build the TRT-Pose engine: `jetson/engines/export_trtpose.py` on a PC, then `make engines`.
