"""Teleop demo logging and the LeRobot v2.1 export (§11.13, §16 Phase 6)."""
import importlib.util
import json
import os
import shutil
import struct
import subprocess
import time
from types import SimpleNamespace

import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _load(name):
    spec = importlib.util.spec_from_file_location(name, os.path.join(ROOT, "jetson", "tools", name + ".py"))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


T = _load("teleop")
X = _load("lerobot_export")


def test_keys_state_joy():
    assert T.keys_in("w\x1b[Aa ") == ["w", "\x1b[A", "a", " "]
    st = {"pose": {"x": 1.5, "y": -2, "yaw": 0.3}, "v": 0.2, "w": 0.1, "head": {"pan": 0.4, "tilt": -0.1}}
    assert T.state_vec(st) == [1.5, -2.0, 0.3, 0.2, 0.1, 0.4, -0.1, 1]
    assert T.state_vec({"pose": None, "v": 0.1}) == [0, 0, 0, 0.1, 0, 0, 0, 0]
    raw = struct.pack("<IhBB", 5, -32767, 2, 1) + struct.pack("<IhBB", 6, 1, 0x81, 0)
    assert T.joy_events(raw) == [(2, 1, -32767), (1, 0, 1)]


def test_velocity_hold_and_joystick():
    t = T.Teleop.__new__(T.Teleop)
    t.a = SimpleNamespace(v=0.25, w=0.8, record=None)
    t.cmd, t.cmd_t, t.joy, t.scale, t.ep = (0, 0), 0.0, (0.0, 0.0), 1.0, None
    t.on_key("w", 10.0)
    assert t.velocity(10.3) == (0.25, 0.0) and t.velocity(10.7) == (0.0, 0.0)   # released after HOLD_S
    t.on_key("+", 11.0)
    t.on_key("a", 11.0)
    assert t.velocity(11.1) == (0.0, 0.8 * 1.25)
    t.on_joy(2, 1, -32767, 11.2)                                                  # stick forward overrides
    v, w = t.velocity(11.2)
    assert abs(v - 0.25 * 1.25) < 1e-6 and w == 0


def _episode(raw, task, t0, n, fps=10, ok=True):
    ep = T.Episode(str(raw), task, now=t0)
    for i in range(n):
        ep.tick(t0 + i / float(fps), 0.1 * i, -0.2, {"pose": {"x": i, "y": 0, "yaw": 0}, "v": 0.1})
    return ep.close(ok, t=t0 + n / float(fps))


def test_episode_file_and_plan(tmp_path):
    p = _episode(tmp_path, "dock", 1000.0, 5)
    ep = X.load_episode(p)
    assert (ep["task"], ep["fps"], len(ep["rows"]), ep["end"]) == ("dock", 10, 5, 1000.5)
    assert ep["rows"][2]["a"] == [0.2, -0.2] and ep["rows"][2]["s"][0] == 2
    assert _episode(tmp_path, "x", 2000.0, 3, ok=False) is None and len(os.listdir(tmp_path)) == 1
    segs = [(100.0, "a"), (400.0, "b"), (700.0, "c"), (2000.0, "d")]
    assert X.plan_clip(segs, 390.0, 450.0) == (["a", "b"], 290.0)
    assert X.plan_clip(segs, 410.0, 420.0) == (["b"], 10.0)
    assert X.plan_clip(segs, 900.0, 1100.0) is None                 # the recorder was off
    assert X.plan_clip(segs, 50.0, 60.0) is None
    cols = X.frame(ep, 3, 1, 100)
    assert cols["index"].tolist() == [100, 101, 102, 103, 104] and cols["timestamp"][4] == pytest.approx(0.4)
    s = X.stats(cols["action"])
    assert s["count"] == [5] and s["max"] == pytest.approx([0.4, -0.2])
    f = X.features(T.STATE_NAMES, 10, (64, 36))
    assert f["observation.state"]["shape"] == [8] and f["observation.images.front"]["shape"] == [36, 64, 3]


@pytest.mark.skipif(not shutil.which("ffmpeg"), reason="ffmpeg not available")
def test_export_end_to_end(tmp_path, monkeypatch):
    raw, rec, out = tmp_path / "raw", tmp_path / "rec", tmp_path / "ds"
    rec.mkdir()
    t0 = time.mktime((2026, 9, 27, 10, 0, 0, 0, 0, -1))
    for cam in (0, 1):
        for k in range(2):                                          # two 3 s segments per camera
            name = "cam%d_%s.ts" % (cam, time.strftime("%Y%m%d-%H%M%S", time.localtime(t0 + 3 * k)))
            subprocess.run(["ffmpeg", "-loglevel", "error", "-f", "lavfi", "-i",
                            "testsrc=size=160x90:rate=10:duration=3", "-c:v", "libx264", "-g", "10",
                            "-f", "mpegts", str(rec / name)], check=True)
    _episode(raw, "come to the sofa", t0 + 2.0, 20)                 # 2 s spanning the segment boundary
    if importlib.util.find_spec("pyarrow") is None:
        monkeypatch.setattr(X, "write_parquet", lambda cols, dst: os.makedirs(os.path.dirname(dst), exist_ok=True)
                            or open(dst, "wb").close())
    assert X.export(str(raw), str(rec), str(out), size=(64, 36), log=lambda *a: None) == 1
    assert X.export(str(raw), str(rec), str(out), size=(64, 36), log=lambda *a: None) == 0     # idempotent
    info = json.load(open(out / "meta" / "info.json"))
    assert (info["total_episodes"], info["total_frames"], info["total_videos"], info["fps"]) == (1, 20, 2, 10)
    assert os.path.exists(out / "data" / "chunk-000" / "episode_000000.parquet")
    mp4 = out / "videos" / "chunk-000" / "observation.images.head" / "episode_000000.mp4"
    n = subprocess.run(["ffprobe", "-v", "error", "-count_frames", "-select_streams", "v:0", "-show_entries",
                        "stream=nb_read_frames", "-of", "csv=p=0", str(mp4)], stdout=subprocess.PIPE, check=True)
    assert int(n.stdout.decode().strip()) == 20
    st = json.loads(open(out / "meta" / "episodes_stats.jsonl").readline())["stats"]
    img = st["observation.images.head"]
    assert np.array(img["mean"]).shape == (3, 1, 1) and 0 < img["mean"][0][0][0] < 1 and img["count"] == [4]
    assert json.loads(open(out / "meta" / "tasks.jsonl").read()) == {"task_index": 0, "task": "come to the sofa"}
