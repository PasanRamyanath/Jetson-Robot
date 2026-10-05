#!/usr/bin/env python3
"""Teleop demos -> LeRobot v2.1 dataset (§11.13, §16 Phase 6). Runs on a PC or Kaggle: Python 3.8+, numpy, pyarrow
and an ffmpeg binary; never on the Nano.

    rsync -a beni:/ssd/beni/lerobot/raw/ raw/ && rsync -a beni:/ssd/beni/rec/ rec/
    python lerobot_export.py raw rec beni_demos            # appends new episodes; safe to re-run

Each `teleop.py --record` episode (JSONL, 10 Hz state + action) is joined with the matching stretch of vision_core's
H.265 recordings (cam<N>_YYYYmmdd-HHMMSS.ts, 5-minute segments, local time in the name) for both cameras. The clip is
cut from those segments, resampled to the episode fps and re-encoded as H.264 (frame i <-> row i). Output layout:
meta/{info.json, episodes.jsonl, tasks.jsonl, episodes_stats.jsonl}, data/chunk-000/episode_000000.parquet,
videos/chunk-000/observation.images.{head,front}/episode_000000.mp4.
"""
import argparse
import glob
import json
import os
import re
import subprocess
import sys
import time

import numpy as np

CAMS = {0: "head", 1: "front"}
SEG = re.compile(r"cam(\d+)_(\d{8}-\d{6})\.ts$")
SEG_MAX_S = 330.0            # a segment is 300 s; allow for the split landing on the next IDR
CHUNK = 1000
DATA_PATH = "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet"
VIDEO_PATH = "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4"


def load_episode(path):
    """-> {task, start, end, fps, rows: [{t, a, s}], state_names} for a finished episode, else None."""
    with open(path) as f:
        lines = [json.loads(ln) for ln in f if ln.strip()]
    if len(lines) < 3 or "task" not in lines[0] or not lines[-1].get("ok"):
        return None
    h = dict(lines[0])
    h.update(rows=[r for r in lines[1:-1] if "a" in r], end=lines[-1]["end"])
    return h if h["rows"] else None


def segments(rec_dir):
    """-> {cam: [(start_wall, path), ...] sorted}."""
    out = {}
    for p in glob.glob(os.path.join(rec_dir, "**", "cam*_*.ts"), recursive=True):
        m = SEG.search(os.path.basename(p))
        if m:
            t = time.mktime(time.strptime(m.group(2), "%Y%m%d-%H%M%S"))
            out.setdefault(int(m.group(1)), []).append((t, p))
    return {c: sorted(v) for c, v in out.items()}


def plan_clip(segs, start, end):
    """Consecutive segments covering [start, end] -> ([paths], offset into the first), or None if there is a gap."""
    first = max((i for i, (t, _) in enumerate(segs) if t <= start), default=None)
    if first is None:
        return None
    paths, i = [], first
    while i < len(segs):
        t, p = segs[i]
        nxt = segs[i + 1][0] if i + 1 < len(segs) else t + SEG_MAX_S
        if nxt - t > SEG_MAX_S:                             # recorder was off between these two
            nxt = t + SEG_MAX_S
        paths.append(p)
        if nxt >= end:
            return paths, start - segs[first][0]
        if i + 1 >= len(segs) or segs[i + 1][0] > nxt + 1:
            return None
        i += 1
    return None


def cut(paths, offset, frames, fps, size, dst):
    """MPEG-TS segments concatenate byte-wise, so ffmpeg's concat: protocol joins them without a list file."""
    w, h = size
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-ss", "%.3f" % offset, "-i", "concat:" + "|".join(paths),
                    "-vf", "fps=%d,scale=%d:%d,tpad=stop_mode=clone:stop_duration=2" % (fps, w, h),
                    "-frames:v", str(frames), "-an", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-g", "2",
                    "-crf", "23", dst], check=True)


def video_stats(mp4, every=5, size=(32, 18)):
    """Per-channel stats in [0, 1], shaped [3, 1, 1] like LeRobot's image stats, from every Nth frame."""
    w, h = size
    raw = subprocess.run(["ffmpeg", "-loglevel", "error", "-i", mp4, "-vf", "select=not(mod(n\\,%d)),scale=%d:%d"
                          % (every, w, h), "-vsync", "0", "-f", "rawvideo", "-pix_fmt", "rgb24", "-"],
                         check=True, stdout=subprocess.PIPE).stdout
    px = np.frombuffer(raw, np.uint8).reshape(-1, 3).astype(np.float64) / 255.0
    n = len(px) // (w * h)
    if not n:
        return {}, 0
    return {k: [[[float(v)]] for v in f(px, axis=0)] for k, f in
            (("min", np.min), ("max", np.max), ("mean", np.mean), ("std", np.std))}, n


def stats(x):
    x = np.asarray(x, np.float64).reshape(len(x), -1)
    return {"min": x.min(0).tolist(), "max": x.max(0).tolist(), "mean": x.mean(0).tolist(), "std": x.std(0).tolist(),
            "count": [len(x)]}


def frame(ep, ep_index, task_index, first_index):
    """Episode -> column arrays (row i is t = i / fps, the same instant as video frame i)."""
    n = len(ep["rows"])
    return {"observation.state": np.array([r["s"] for r in ep["rows"]], np.float32),
            "action": np.array([r["a"] for r in ep["rows"]], np.float32),
            "timestamp": np.arange(n, dtype=np.float32) / ep["fps"],
            "frame_index": np.arange(n, dtype=np.int64),
            "episode_index": np.full(n, ep_index, np.int64),
            "index": np.arange(first_index, first_index + n, dtype=np.int64),
            "task_index": np.full(n, task_index, np.int64)}


def features(state_names, fps, size):
    w, h = size
    f = {"observation.state": {"dtype": "float32", "shape": [len(state_names)], "names": list(state_names)},
         "action": {"dtype": "float32", "shape": [2], "names": ["v", "w"]}}
    for name in CAMS.values():
        f["observation.images." + name] = {
            "dtype": "video", "shape": [h, w, 3], "names": ["height", "width", "channels"],
            "info": {"video.fps": fps, "video.height": h, "video.width": w, "video.channels": 3, "video.codec": "h264",
                     "video.pix_fmt": "yuv420p", "video.is_depth_map": False, "has_audio": False}}
    for k in ("timestamp",):
        f[k] = {"dtype": "float32", "shape": [1], "names": None}
    for k in ("frame_index", "episode_index", "index", "task_index"):
        f[k] = {"dtype": "int64", "shape": [1], "names": None}
    return f


def write_parquet(cols, dst):
    import pyarrow as pa
    import pyarrow.parquet as pq
    arrs = {}
    for k, v in cols.items():
        if v.ndim == 2:
            arrs[k] = pa.FixedSizeListArray.from_arrays(pa.array(v.ravel()), v.shape[1])
        else:
            arrs[k] = pa.array(v)
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    pq.write_table(pa.table(arrs), dst)


def _json(path, default=None):
    if not os.path.exists(path):
        return default
    with open(path) as f:
        return json.load(f)


def _jsonl(path):
    if not os.path.exists(path):
        return []
    with open(path) as f:
        return [json.loads(ln) for ln in f if ln.strip()]


def export(raw_dir, rec_dir, out, size=(640, 360), robot_type="beni_diffdrive", log=print):
    meta = os.path.join(out, "meta")
    os.makedirs(meta, exist_ok=True)
    info_p = os.path.join(meta, "info.json")
    info = _json(info_p)
    episodes = _jsonl(os.path.join(meta, "episodes.jsonl"))
    tasks = {t["task"]: t["task_index"] for t in _jsonl(os.path.join(meta, "tasks.jsonl"))}
    src_p = os.path.join(meta, "beni_sources.json")
    done = set(_json(src_p, []))
    segs = segments(rec_dir)
    total = info["total_frames"] if info else 0
    added = 0
    for path in sorted(glob.glob(os.path.join(raw_dir, "ep_*.jsonl"))):
        name = os.path.basename(path)
        ep = None if name in done else load_episode(path)
        if ep is None:
            continue
        if info and info["fps"] != ep["fps"]:
            log("skip %s: fps %s != dataset fps %s" % (name, ep["fps"], info["fps"]))
            continue
        n, t0 = len(ep["rows"]), ep["rows"][0]["t"]
        plans = {c: plan_clip(segs.get(c, []), t0, t0 + n / float(ep["fps"])) for c in CAMS}
        if not all(plans.values()):
            log("skip %s: no recording covers it for cam %s" % (name, [c for c, p in plans.items() if not p]))
            continue
        idx, chunk = len(episodes), len(episodes) // CHUNK
        ti = tasks.setdefault(ep["task"], len(tasks))
        cols = frame(ep, idx, ti, total)
        st = {k: stats(v) for k, v in cols.items()}
        for c, cam in CAMS.items():
            key = "observation.images." + cam
            mp4 = os.path.join(out, VIDEO_PATH.format(episode_chunk=chunk, video_key=key, episode_index=idx))
            cut(plans[c][0], plans[c][1], n, ep["fps"], size, mp4)
            s, k = video_stats(mp4)
            st[key] = dict(s, count=[k])
        write_parquet(cols, os.path.join(out, DATA_PATH.format(episode_chunk=chunk, episode_index=idx)))
        episodes.append({"episode_index": idx, "tasks": [ep["task"]], "length": n})
        with open(os.path.join(meta, "episodes_stats.jsonl"), "a") as f:
            f.write(json.dumps({"episode_index": idx, "stats": st}) + "\n")
        done.add(name)
        total += n
        added += 1
        info = info or {"fps": ep["fps"], "features": features(ep.get("state_names") or [], ep["fps"], size)}
        log("episode %d <- %s (%d frames, '%s')" % (idx, name, n, ep["task"]))
    if not added:
        log("nothing new")
        return 0
    with open(os.path.join(meta, "episodes.jsonl"), "w") as f:
        f.writelines(json.dumps(e) + "\n" for e in episodes)
    with open(os.path.join(meta, "tasks.jsonl"), "w") as f:
        f.writelines(json.dumps({"task_index": i, "task": t}) + "\n" for t, i in sorted(tasks.items(),
                                                                                         key=lambda x: x[1]))
    info.update(codebase_version="v2.1", robot_type=robot_type, total_episodes=len(episodes), total_frames=total,
                total_tasks=len(tasks), total_videos=len(episodes) * len(CAMS),
                total_chunks=(len(episodes) - 1) // CHUNK + 1, chunks_size=CHUNK,
                splits={"train": "0:%d" % len(episodes)}, data_path=DATA_PATH, video_path=VIDEO_PATH)
    with open(info_p, "w") as f:
        json.dump(info, f, indent=2)
    with open(src_p, "w") as f:
        json.dump(sorted(done), f)
    return added


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("raw")
    p.add_argument("rec")
    p.add_argument("out")
    p.add_argument("--size", default="640x360", help="video WxH")
    a = p.parse_args(argv)
    w, h = (int(x) for x in a.size.split("x"))
    export(a.raw, a.rec, a.out, (w, h))
    return 0


if __name__ == "__main__":
    sys.exit(main())
