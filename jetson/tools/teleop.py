#!/usr/bin/env python3
"""Keyboard / joystick teleop with LeRobot-style demo logging (§16 Phases 1 and 6, §11.13). Host Python 3.6.

    teleop.py                        drive over SSH: w/s a/d (or arrows), q/e arcs, space stop, +/- speed, Ctrl-C quit
    teleop.py --record "dock at the charger"
                                     r: start/end a demo episode, x: discard it (joystick: button 0 / button 1)
    teleop.py --joy /dev/input/js0   left stick drives (overrides the keyboard while deflected)

Commands go to the ROS bridge as {op: drive} at 10 Hz (base mux: teleop wins for 0.5 s). A demo is one JSONL file in
--out: a header {task, start, fps, cams}, then one {t, a: [v, w], s: [x, y, yaw, v, w, pan, tilt, pose_ok]} per tick,
then {end, ok}. Frames are not copied: vision_core already records both cameras, and `lerobot_export.py` (PC/Kaggle)
cuts them from those segments into a LeRobot v2.1 dataset. Keep recording on (no --no-rec) while collecting demos.
"""
import argparse
import json
import os
import select
import struct
import sys
import time

import zmq

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "shared"))
from beni_common import schemas as S  # noqa: E402

FPS = 10
HOLD_S = 0.6                # a key keeps driving this long (terminal auto-repeat refreshes it while held)
STOP_S = 0.5                # send zeros this long after release, then go quiet so nav/behaviour get the base back
DEAD = 0.12
KEYS = {"w": (1, 0), "s": (-1, 0), "a": (0, 1), "d": (0, -1), "q": (1, 1), "e": (1, -1),
        "\x1b[A": (1, 0), "\x1b[B": (-1, 0), "\x1b[D": (0, 1), "\x1b[C": (0, -1)}
STATE_NAMES = ["x", "y", "yaw", "v", "w", "pan", "tilt", "pose_ok"]


def keys_in(buf):
    """Split a terminal read into keys (arrow keys are 3-byte escape sequences)."""
    out, i = [], 0
    while i < len(buf):
        if buf[i] == "\x1b" and buf[i + 1:i + 2] == "[" and i + 2 < len(buf):
            out.append(buf[i:i + 3])
            i += 3
        else:
            out.append(buf[i])
            i += 1
    return out


def state_vec(st):
    """robot_state msg -> the observation.state vector (0s and pose_ok=0 while the pose is unknown)."""
    st = st or {}
    p, h = st.get("pose") or {}, st.get("head") or {}
    return [round(float(v), 4) for v in (p.get("x", 0), p.get("y", 0), p.get("yaw", 0), st.get("v", 0),
                                          st.get("w", 0), h.get("pan", 0), h.get("tilt", 0), 1 if p else 0)]


def joy_events(buf):
    """Linux joystick API records (u32 ms, s16 value, u8 type, u8 number) -> [(type & 3, number, value)]."""
    return [(t & 3, n, v) for _, v, t, n in (struct.unpack("<IhBB", buf[i:i + 8]) for i in range(0, len(buf) - 7, 8))]


class Episode(object):
    def __init__(self, out_dir, task, now=None):
        now = now or time.time()
        if not os.path.isdir(out_dir):
            os.makedirs(out_dir)
        self.path = os.path.join(out_dir, "ep_%s.jsonl" % time.strftime("%Y%m%d-%H%M%S", time.localtime(now)))
        self.f = open(self.path, "w")
        self.n = 0
        self._w({"task": task, "start": now, "fps": FPS, "cams": [0, 1], "state_names": STATE_NAMES, "v": 1})

    def _w(self, obj):
        self.f.write(json.dumps(obj, separators=(",", ":")) + "\n")

    def tick(self, t, v, w, st):
        self._w({"t": round(t, 3), "a": [round(v, 3), round(w, 3)], "s": state_vec(st)})
        self.n += 1

    def close(self, keep, t=None):
        if keep:
            self._w({"end": round(t or time.time(), 3), "ok": True})
        self.f.close()
        if not keep:
            os.remove(self.path)
        return self.path if keep else None


class Teleop(object):
    def __init__(self, a):
        self.a, self.ctx = a, zmq.Context.instance()
        self.req = self._req()
        self.sub = self.ctx.socket(zmq.SUB)
        self.sub.setsockopt(zmq.SUBSCRIBE, S.T_STATE)
        self.sub.setsockopt(zmq.RCVHWM, 4)
        self.sub.connect(S.EP["robot_state"])
        self.state, self.ep = None, None
        self.cmd, self.cmd_t, self.joy = (0, 0), 0.0, (0.0, 0.0)
        self.scale, self.last_move = 1.0, -1e9

    def _req(self):
        s = self.ctx.socket(zmq.REQ)
        s.setsockopt(zmq.LINGER, 0)
        s.setsockopt(zmq.RCVTIMEO, 300)
        s.connect(S.EP["robot_cmd"])
        return s

    def send(self, v, w):
        try:
            self.req.send(S.pack({"op": "drive", "v": v, "w": w}))
            r = S.unpack(self.req.recv())
            return bool(r.get("ok"))
        except zmq.Again:                                   # lazy pirate: a REQ that timed out must be replaced
            self.req.close()
            self.req = self._req()
            return False

    def drain_state(self):
        while True:
            try:
                _, raw = self.sub.recv_multipart(zmq.NOBLOCK)
            except zmq.Again:
                return
            self.state = S.unpack(raw)

    def on_key(self, k, now):
        if k in KEYS:
            self.cmd, self.cmd_t = KEYS[k], now
        elif k == " ":
            self.cmd, self.cmd_t = (0, 0), 0.0
        elif k in "+=":
            self.scale = min(2.0, self.scale * 1.25)
        elif k in "-_":
            self.scale = max(0.25, self.scale / 1.25)
        elif k in "rR":
            self.toggle()
        elif k in "xX":
            self.discard()

    def on_joy(self, typ, n, v, now):
        if typ == 2 and n in (0, 1):
            x, y = self.joy
            self.joy = (v / 32767.0, y) if n == 0 else (x, v / 32767.0)
        elif typ == 1 and v == 1 and n in (0, 1):
            (self.toggle if n == 0 else self.discard)()

    def toggle(self):
        if not self.a.record:
            print("\r(not recording: start with --record TASK)")
        elif self.ep is None:
            self.ep = Episode(self.a.out, self.a.record)
            print("\r[rec] %s" % self.ep.path)
        else:
            n, path = self.ep.n, self.ep.close(True)
            self.ep = None
            print("\r[saved] %s (%d ticks, %.1f s)" % (path, n, n / float(FPS)))

    def discard(self):
        if self.ep is not None:
            self.ep.close(False)
            self.ep = None
            print("\r[discarded]")

    def velocity(self, now):
        jx, jy = self.joy
        if abs(jx) > DEAD or abs(jy) > DEAD:
            return -jy * self.a.v * self.scale, -jx * self.a.w * self.scale
        if now - self.cmd_t < HOLD_S:
            return self.cmd[0] * self.a.v * self.scale, self.cmd[1] * self.a.w * self.scale
        return 0.0, 0.0

    def tick(self, now, wall):
        self.drain_state()
        v, w = self.velocity(now)
        if v or w:
            self.last_move = now
        if v or w or now - self.last_move < STOP_S:
            self.send(v, w)
        if self.ep is not None:
            self.ep.tick(wall, v, w, self.state)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--record", metavar="TASK", help="task text for demo episodes (enables r / x)")
    p.add_argument("--out", default=os.environ.get("BENI_DEMO_DIR", "/ssd/beni/lerobot/raw"))
    p.add_argument("--joy", help="joystick device, e.g. /dev/input/js0")
    p.add_argument("--v", type=float, default=0.25, help="m/s at scale 1")
    p.add_argument("--w", type=float, default=0.8, help="rad/s at scale 1")
    a = p.parse_args(argv)
    import termios
    import tty
    t = Teleop(a)
    fd = sys.stdin.fileno()
    old = termios.tcgetattr(fd)
    joy = os.open(a.joy, os.O_RDONLY | os.O_NONBLOCK) if a.joy else None
    print(__doc__.split("\n\n")[1] if not a.record else "recording '%s': r start/end, x discard" % a.record)
    try:
        tty.setcbreak(fd)
        nxt = time.monotonic()
        while True:
            now = time.monotonic()
            rd, _, _ = select.select([fd] + ([joy] if joy is not None else []), [], [], max(0.0, nxt - now))
            now = time.monotonic()
            if fd in rd:
                for k in keys_in(os.read(fd, 64).decode("utf-8", "ignore")):
                    t.on_key(k, now)
            if joy is not None and joy in rd:
                for ev in joy_events(os.read(joy, 512)):
                    t.on_joy(ev[0], ev[1], ev[2], now)
            if now >= nxt:
                t.tick(now, time.time())
                nxt += 1.0 / FPS
                if nxt < now:                               # fell behind (SSH hiccup): don't burst
                    nxt = now + 1.0 / FPS
    except KeyboardInterrupt:
        pass
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old)
        if t.ep is not None:
            t.toggle()
        t.send(0.0, 0.0)
    return 0


if __name__ == "__main__":
    sys.exit(main())
