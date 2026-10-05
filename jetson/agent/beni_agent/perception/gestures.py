"""Gestures (§4 item 39): wave and point from TRT-Pose keypoints, which vision_core computes only on demand.

`watch(cam, secs)` asks vision_core ({op: pose}) for keypoints on a camera for a while (around a voice turn, or when
someone arrives); the `pose` topic then carries 18 keypoints per person track at 5 Hz. Plain geometry on the last two
seconds of each track, all in 512x288 detector pixels, scaled by shoulder width:
- wave: a wrist above its shoulder that swings side to side (two or more reversals);
- point: a straight arm held out sideways for 0.6 s -> "left" / "right" as the robot sees it.
"""
import collections
import logging
import math
import time

from beni_common import schemas as S

from ..util import spawn

log = logging.getLogger("gestures")

NOSE, L_SH, R_SH, L_EL, R_EL, L_WR, R_WR, NECK = 0, 5, 6, 7, 8, 9, 10, 17
ARMS = ((L_SH, L_EL, L_WR), (R_SH, R_EL, R_WR))
CONF = 0.15
WIN_S = 2.0
WAVE_MIN_FRAMES = 4
WAVE_SWING = 0.35           # x swing between reversals, in shoulder widths
POINT_HOLD_S = 0.6
POINT_ELBOW_DEG = 150
POINT_REACH = 1.1           # wrist-shoulder x distance, in shoulder widths
COOLDOWN_S = 4.0
DISABLED_S = 600


def _pt(kps, i):
    x, y, c = kps[3 * i:3 * i + 3]
    return (x, y) if c >= CONF else None


def _scale(kps):
    a, b = _pt(kps, L_SH), _pt(kps, R_SH)
    return max(abs(a[0] - b[0]), 8.0) if a and b else None


def raised(kps, arm):
    sh, _, wr = (_pt(kps, i) for i in arm)
    s = _scale(kps)
    return bool(sh and wr and s) and wr[1] < sh[1] - 0.2 * s


def pointing(kps, arm):
    """-> -1 (image left), +1 (image right) or 0."""
    sh, el, wr = (_pt(kps, i) for i in arm)
    s = _scale(kps)
    if not (sh and el and wr and s):
        return 0
    dx, dy = wr[0] - sh[0], wr[1] - sh[1]
    if abs(dx) < POINT_REACH * s or abs(dy) > abs(dx):          # not out sideways, or more up/down than across
        return 0
    a = (sh[0] - el[0], sh[1] - el[1])
    b = (wr[0] - el[0], wr[1] - el[1])
    na, nb = math.hypot(*a), math.hypot(*b)
    if na == 0 or nb == 0:
        return 0
    cos = max(-1.0, min(1.0, (a[0] * b[0] + a[1] * b[1]) / (na * nb)))
    if math.degrees(math.acos(cos)) < POINT_ELBOW_DEG:
        return 0
    return -1 if dx < 0 else 1


def reversals(xs, swing):
    """Direction changes of a 1-D path, ignoring wiggles smaller than `swing`."""
    n, d, ext, lo, hi = 0, 0, xs[0], xs[0], xs[0]
    for x in xs[1:]:
        if d == 0:                                  # no direction yet: wait for the first full swing
            lo, hi = min(lo, x), max(hi, x)
            if hi - lo >= swing:
                d, ext = (1 if x == hi else -1), x
        elif (x - ext) * d > 0:
            ext = x
        elif abs(x - ext) >= swing:
            n, d, ext = n + 1, -d, x
    return n


class Gestures:
    def __init__(self, request, on_gesture, clock=time.time):
        self.request, self.on_gesture, self.clock = request, on_gesture, clock
        self.hist = {}              # (cam, tid) -> deque[(t, kps)]
        self.last = {}              # (cam, tid, name) -> t
        self.until = {}             # cam -> t the current watch ends
        self.off_until = 0.0

    # ------------------------------------------------------------------ demand
    async def watch(self, cam=0, secs=15.0):
        now = self.clock()
        if now < self.off_until or self.until.get(cam, 0) > now + secs / 2:
            return False
        r = await self.request(S.EP["vision_ctrl"], {"op": "pose", "cam": cam, "secs": secs})
        if not r.get("ok"):
            if r.get("err") in ("pose off", "unknown op"):          # no engine, or the DeepStream path
                self.off_until = now + DISABLED_S
            return False
        self.until[cam] = now + secs
        return True

    # ------------------------------------------------------------------ keypoints
    def on_msg(self, topic, msg):
        if topic != S.T_POSE:
            return
        now, cam = self.clock(), msg.get("cam", 0)
        for p in msg.get("people", ()):
            kps = p.get("kps") or ()
            if len(kps) != 54:
                continue
            key = (cam, p.get("tid"))
            h = self.hist.setdefault(key, collections.deque())
            h.append((now, kps))
            while h and now - h[0][0] > WIN_S:
                h.popleft()
            self._check(key, h, now)
        if len(self.hist) > 32:
            self.hist = {k: h for k, h in self.hist.items() if h and now - h[-1][0] < WIN_S}

    def _check(self, key, h, now):
        for arm in ARMS:
            up = [(kps[3 * arm[2]], _scale(kps)) for _, kps in h if raised(kps, arm)]
            if len(up) >= WAVE_MIN_FRAMES:
                swing = WAVE_SWING * sorted(s for _, s in up)[len(up) // 2]
                if reversals([x for x, _ in up], swing) >= 2 and self._fire(key, "wave", now, {}):
                    return
            held = [pointing(kps, arm) for t, kps in h if now - t <= POINT_HOLD_S]
            span = now - next((t for t, _ in h if now - t <= POINT_HOLD_S), now)
            if len(held) >= 3 and span >= POINT_HOLD_S - 0.25 and held[0] and held.count(held[0]) == len(held):
                wr = h[-1][1][3 * arm[2]]
                if self._fire(key, "point", now, {"dir": "left" if held[0] < 0 else "right", "x": round(wr / 512, 3)}):
                    return

    def _fire(self, key, name, now, extra):
        k = key + (name,)
        if now - self.last.get(k, -1e9) < COOLDOWN_S:
            return False
        self.last[k] = now
        log.info("%s cam%d track %s %s", name, key[0], key[1], extra)
        self.on_gesture(name, key[0], key[1], extra)
        return True

    def watch_soon(self, cam=0, secs=15.0):
        """Fire-and-forget watch() from sync callbacks."""
        spawn(self._safe_watch(cam, secs))

    async def _safe_watch(self, cam, secs):
        try:
            await self.watch(cam, secs)
        except Exception as e:
            log.debug("pose watch: %s", e)
