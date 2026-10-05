#!/usr/bin/env python3
"""Procedural expression clips for beni_face (§3.11).

800x480 H.264 Annex-B, IDR at frame 0, AUD per frame, no B-frames.

Runs on the dev PC (numpy + the x264 CLI, or ffmpeg with libx264), not on the Nano. For each clip it writes:
  <name>.h264   the stream beni_face feeds to NVDEC, looping it and cross-cutting to another clip at frame 0
  <name>.eyes   (only for clips whose pupils live on overlay plane 1) eye geometry + per-frame eyelid openness,
                so gaze-tracked pupils stay inside the eye and close with each blink

    python make_clips.py [--out clips] [--only idle_blink,happy] [--preview]
    rsync -a clips/ beni@beni.local:/ssd/face/
"""
import argparse
import math
import os
import shutil
import subprocess
import sys

import numpy as np

W, H, FPS = 800, 480, 30
L, R = (250.0, 235.0), (550.0, 235.0)      # eye centres
HW, HH, PR = 105.0, 118.0, 46.0            # eye half-width / half-height, pupil radius (overlay)
BG = (6, 8, 14)
EYE = (215, 238, 255)
PUPIL = (18, 20, 32)
ACCENT = (120, 200, 255)

YY, XX = np.mgrid[0:H, 0:W].astype(np.float32) + 0.5


# ---------------------------------------------------------------------------- drawing (anti-aliased coverage masks)
def ellipse(cx, cy, rx, ry, rot=0.0):
    if rx < 0.5 or ry < 0.5:
        return np.zeros((H, W), np.float32)
    dx, dy = XX - cx, YY - cy
    if rot:
        c, s = math.cos(rot), math.sin(rot)
        dx, dy = c * dx + s * dy, -s * dx + c * dy
    q = np.sqrt((dx / rx) ** 2 + (dy / ry) ** 2)
    return np.clip((1.0 - q) * min(rx, ry) + 0.5, 0, 1)


def capsule(x0, y0, x1, y1, w):
    dx, dy = x1 - x0, y1 - y0
    t = np.clip(((XX - x0) * dx + (YY - y0) * dy) / max(1e-6, dx * dx + dy * dy), 0, 1)
    return np.clip(w / 2 + 0.5 - np.hypot(XX - x0 - t * dx, YY - y0 - t * dy), 0, 1)


def arc(cx, cy, r, a0, a1, w):
    """Thick arc from angle a0 to a1 (radians, y down), round caps."""
    m = np.zeros((H, W), np.float32)
    n = max(8, int(abs(a1 - a0) * r / 6))
    pts = [(cx + r * math.cos(a0 + (a1 - a0) * k / n), cy + r * math.sin(a0 + (a1 - a0) * k / n)) for k in range(n + 1)]
    for (xa, ya), (xb, yb) in zip(pts, pts[1:]):
        m = np.maximum(m, capsule(xa, ya, xb, yb, w))
    return m


def triangle(pts):
    """Anti-aliased convex triangle (clockwise or not) from the min signed edge distance."""
    d = None
    cross = (pts[1][0] - pts[0][0]) * (pts[2][1] - pts[0][1]) - (pts[1][1] - pts[0][1]) * (pts[2][0] - pts[0][0])
    sign = 1 if cross > 0 else -1
    for (xa, ya), (xb, yb) in zip(pts, pts[1:] + pts[:1]):
        ex, ey = xb - xa, yb - ya
        e = sign * ((XX - xa) * ey - (YY - ya) * ex) / -math.hypot(ex, ey)
        d = e if d is None else np.minimum(d, e)
    return np.clip(d + 0.5, 0, 1)


def heart(cx, cy, s):
    """Two lobes + a point: the lobes' outer tangents meet the triangle's sides."""
    r, a = 0.52 * s, math.radians(45)
    lx, rx, ly = cx - 0.5 * s, cx + 0.5 * s, cy - 0.3 * s
    t = [(lx - r * math.cos(a), ly + r * math.sin(a)), (rx + r * math.cos(a), ly + r * math.sin(a)),
         (cx, cy + 0.95 * s)]
    return np.maximum(np.maximum(ellipse(lx, ly, r, r), ellipse(rx, ly, r, r)),
                      np.maximum(triangle(t), ellipse(cx, ly + 0.2 * s, 0.45 * s, 0.45 * s)))


class Frame:
    def __init__(self):
        self.rgb = np.empty((H, W, 3), np.float32)
        self.rgb[:] = BG

    def paint(self, mask, color, alpha=1.0):
        a = (mask * alpha)[..., None]
        self.rgb = self.rgb * (1 - a) + np.asarray(color, np.float32) * a

    def eye(self, c, open_=1.0, sx=1.0, sy=1.0, pupil=None, glow=True):
        """Sclera ellipse scaled vertically by eyelid openness; optional in-clip pupil (dx, dy, r)."""
        cx, cy = c
        hh = HH * sy * max(open_, 0.0)
        if glow:
            self.paint(ellipse(cx, cy, HW * sx + 14, hh + 14), ACCENT, 0.12)
        if open_ < 0.06:                      # closed: a lid line
            self.paint(capsule(cx - HW * sx, cy, cx + HW * sx, cy, 10), EYE)
            return
        m = ellipse(cx, cy, HW * sx, hh)
        self.paint(m, EYE)
        if pupil is not None:
            dx, dy, r = pupil
            p = np.minimum(ellipse(cx + dx, cy + dy, r, r), m)
            self.paint(p, PUPIL)
            self.paint(np.minimum(ellipse(cx + dx - r * .35, cy + dy - r * .4, r * .28, r * .28), m), (255, 255, 255))

    def yuv420(self):
        """BT.601 limited range, the default the NVDEC -> DC path assumes for untagged streams."""
        r, g, b = self.rgb[..., 0], self.rgb[..., 1], self.rgb[..., 2]
        y = 16 + 0.257 * r + 0.504 * g + 0.098 * b
        u = 128 - 0.148 * r - 0.291 * g + 0.439 * b
        v = 128 + 0.439 * r - 0.368 * g - 0.071 * b
        sub = lambda p: p.reshape(H // 2, 2, W // 2, 2).mean(axis=(1, 3))     # noqa: E731
        return b"".join(np.clip(p + 0.5, 0, 255).astype(np.uint8).tobytes() for p in (y, sub(u), sub(v)))


# ---------------------------------------------------------------------------- lid curves
def blink(n, at, dur=7):
    """Openness per frame with one blink starting at frame `at` (close fast, open slower)."""
    out = []
    for i in range(n):
        k = i - at
        if 0 <= k < dur:
            h = dur * 0.4
            out.append(1 - k / h if k < h else (k - h) / (dur - h))
        else:
            out.append(1.0)
    return [max(0.0, min(1.0, v)) for v in out]


def ease(t):
    return 0.5 - 0.5 * math.cos(math.pi * t)


# ---------------------------------------------------------------------------- clips: name -> (frames, overlay pupils?)
def c_idle_blink():
    n = FPS * 4
    lid = blink(n, int(FPS * 2.6))
    return n, lid, lambda f, i: [f.eye(c, lid[i]) for c in (L, R)]


def c_talking():
    n = FPS * 2
    lid = [1.0] * n

    def draw(f, i):
        b = 6 * abs(math.sin(2 * math.pi * i / (n / 4)))                 # gentle bounce, 4 per loop
        for c in (L, R):
            f.eye((c[0], c[1] - b), 1.0, sy=1.0 - b / 200)
    return n, lid, draw


def c_listening():
    n = FPS * 2
    lid = [1.0] * n

    def draw(f, i):
        k = 0.5 + 0.5 * math.sin(2 * math.pi * i / n)
        for c in (L, R):
            f.paint(ellipse(c[0], c[1], HW + 26 + 6 * k, HH + 26 + 6 * k), ACCENT, 0.10 + 0.12 * k)   # attentive glow
            f.eye(c, 1.0, sx=1.04, sy=1.04, glow=False)
    return n, lid, draw


def c_surprised():
    n = FPS * 2
    lid = [1.0] * n
    return n, lid, lambda f, i: [f.eye(c, 1.0, sx=1.12, sy=1.12) for c in (L, R)]


def c_curious():
    n = FPS * 3
    lid = blink(n, int(FPS * 2.2))

    def draw(f, i):
        f.eye(L, lid[i], sx=0.92, sy=0.8)
        f.eye(R, lid[i], sx=1.08, sy=1.1)
    return n, lid, draw


def c_sleepy():
    n = FPS * 4
    lid = [0.32 + 0.1 * math.sin(2 * math.pi * i / n) for i in range(n)]
    for i in range(int(FPS * 2.0), int(FPS * 3.0)):                   # slow heavy blink
        lid[i] *= 1 - math.sin(math.pi * (i - FPS * 2.0) / FPS)
    return n, lid, lambda f, i: [f.eye((c[0], c[1] + 18), lid[i]) for c in (L, R)]


def c_closed():
    """§12.3 camera privacy mode: lids shut, dim glow, no pupils (no sidecar)."""
    return FPS, None, lambda f, i: [f.eye(c, 0.0) for c in (L, R)]


def c_thinking():
    n = FPS * 3

    def draw(f, i):
        t = ease(min(1.0, i / (FPS * 0.5))) if i < n / 2 else ease(min(1.0, (n - i) / (FPS * 0.5)))
        for c in (L, R):
            f.eye(c, 0.75, pupil=(40 * t, -45 * t, PR))
        for k in range(3):                                            # thought dots
            on = (i // (FPS // 3)) % 4 > k
            f.paint(ellipse(640 + 34 * k, 90 - 14 * k, 9 + 2 * k, 9 + 2 * k), ACCENT, 0.9 if on else 0.2)
    return n, None, draw


def c_happy():
    n = FPS * 2

    def draw(f, i):
        b = 4 * math.sin(2 * math.pi * i / n)
        for c in (L, R):
            f.paint(arc(c[0], c[1] + 40 + b, 78, math.pi * 1.15, math.pi * 1.85, 26), EYE)
        for c in (L, R):
            f.paint(ellipse(c[0], c[1] + 120, 46, 18), (255, 120, 150), 0.35)   # blush
    return n, None, draw


def c_sad():
    n = FPS * 3

    def draw(f, i):
        for c, s in ((L, 1), (R, -1)):
            f.eye((c[0], c[1] + 20), 0.62, pupil=(0, 20, PR))
            f.paint(capsule(c[0] - 95, c[1] - 58 + 36 * s, c[0] + 95, c[1] - 58 - 36 * s, 16), EYE)   # droopy brows
        y = 280 + (i % n) * 3.0
        f.paint(ellipse(R[0] + 70, y, 10, 15), ACCENT, 0.8 if y < 440 else 0)
    return n, None, draw


def c_angry():
    n = FPS * 2

    def draw(f, i):
        for c, s in ((L, 1), (R, -1)):
            f.eye(c, 0.7, pupil=(0, 10, PR))
            edge = YY - (c[1] - 60 + 0.35 * s * (XX - c[0]))                       # slanted lid, anti-aliased
            f.paint(np.clip(0.5 - edge, 0, 1) * (np.abs(XX - c[0]) < HW + 20), BG)
            f.paint(capsule(c[0] - 100, c[1] - 92 - 35 * s, c[0] + 100, c[1] - 92 + 35 * s, 20), (255, 90, 70))
    return n, None, draw


def c_love():
    n = FPS * 2

    def draw(f, i):
        s = 95 * (1 + 0.08 * math.sin(2 * math.pi * i / (n / 2)))      # heartbeat
        for c in (L, R):
            f.paint(heart(c[0], c[1], s), (255, 80, 120))
    return n, None, draw


def c_wink():
    n = FPS * 2

    def draw(f, i):
        f.eye(L, 1.0, pupil=(0, 0, PR))
        f.paint(arc(R[0], R[1] + 40, 78, math.pi * 1.15, math.pi * 1.85, 26), EYE)
    return n, None, draw


CLIPS = {k[2:]: v for k, v in globals().items() if k.startswith("c_")}


# ---------------------------------------------------------------------------- encoding
def encoder_cmd(n, out):
    if shutil.which("x264"):
        return ["x264", "--input-res", "%dx%d" % (W, H), "--input-csp", "i420", "--fps", str(FPS), "--demuxer", "raw",
                "--profile", "high", "--level", "4.0", "--preset", "veryslow", "--tune", "animation", "--crf", "18",
                "--keyint", str(n), "--min-keyint", str(n), "--no-scenecut", "--bframes", "0", "--aud",
                "--colormatrix", "bt470bg", "--quiet", "-o", out, "-"]
    if shutil.which("ffmpeg"):
        return ["ffmpeg", "-loglevel", "error", "-y", "-f", "rawvideo", "-pix_fmt", "yuv420p", "-s", "%dx%d" % (W, H),
                "-r", str(FPS), "-i", "-", "-c:v", "libx264", "-profile:v", "high", "-preset", "veryslow",
                "-tune", "animation", "-crf", "18", "-x264-params",
                "keyint=%d:min-keyint=%d:scenecut=0:bframes=0:aud=1" % (n, n), "-colorspace", "bt470bg",
                "-f", "h264", out]
    sys.exit("need x264 or ffmpeg (with libx264) on PATH")


def render(name, out_dir, preview=False):
    n, lid, draw = CLIPS[name]()
    path = os.path.join(out_dir, name + ".h264")
    p = subprocess.Popen(encoder_cmd(n, path), stdin=subprocess.PIPE)
    for i in range(n):
        f = Frame()
        draw(f, i)
        p.stdin.write(f.yuv420())
        if preview and i == n // 3:
            _png(f, os.path.join(out_dir, name + ".png"))
    p.stdin.close()
    if p.wait():
        sys.exit("%s: encoder failed" % name)
    side = os.path.join(out_dir, name + ".eyes")
    if lid is not None:
        with open(side, "w") as fh:
            fh.write("fps %d\neyes %.1f %.1f %.1f %.1f %.1f %.1f %.1f\nlid %s\n" % (
                FPS, L[0], L[1], R[0], R[1], HW, HH, PR, " ".join("%.3f" % v for v in lid)))
    elif os.path.exists(side):
        os.remove(side)
    print("%-11s %3d frames %6.1f KB%s" % (name, n, os.path.getsize(path) / 1024., " + eyes" if lid else ""))


def _png(f, path):
    import zlib
    raw = np.clip(f.rgb + 0.5, 0, 255).astype(np.uint8)
    rows = b"".join(b"\0" + raw[y].tobytes() for y in range(H))

    def chunk(t, d):
        return len(d).to_bytes(4, "big") + t + d + zlib.crc32(t + d).to_bytes(4, "big")
    with open(path, "wb") as fh:
        fh.write(b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", W.to_bytes(4, "big") + H.to_bytes(4, "big") + b"\x08\x02\0\0\0")
                 + chunk(b"IDAT", zlib.compress(rows, 6)) + chunk(b"IEND", b""))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "clips"))
    ap.add_argument("--only", default="", help="comma-separated subset of: " + ", ".join(sorted(CLIPS)))
    ap.add_argument("--preview", action="store_true", help="also write one PNG still per clip")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    for name in (a.only.split(",") if a.only else sorted(CLIPS)):
        render(name, a.out, a.preview)


if __name__ == "__main__":
    main()
