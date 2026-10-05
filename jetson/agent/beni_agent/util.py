"""Small process utilities: sd_notify (no systemd python dep), per-turn latency log, logging setup, spawn."""
import asyncio
import json
import logging
import os
import socket
import time


def sd_notify(msg):
    """Send READY=1 / WATCHDOG=1 / STATUS=... to systemd if NOTIFY_SOCKET is set."""
    addr = os.environ.get("NOTIFY_SOCKET")
    if not addr:
        return False
    if addr[0] == "@":
        addr = "\0" + addr[1:]
    try:
        with socket.socket(socket.AF_UNIX, socket.SOCK_DGRAM) as s:
            s.sendto(msg.encode(), addr)
        return True
    except OSError:
        return False


_BG = set()


def spawn(coro):
    """ensure_future that keeps a strong reference (the loop holds only weak ones) and logs a failure."""
    t = asyncio.ensure_future(coro)
    _BG.add(t)
    t.add_done_callback(_done)
    return t


def _done(t):
    _BG.discard(t)
    if not t.cancelled() and t.exception() is not None:
        logging.getLogger("task").error("background task failed", exc_info=t.exception())


class TurnLog:
    """Appends one JSON line per voice turn with stage timestamps (read by tools/voice_latency_report.py)."""

    def __init__(self, path):
        self.path = path
        self.cur = {}
        if path:
            os.makedirs(os.path.dirname(path), exist_ok=True)

    def begin(self, turn, **kw):
        self.flush()
        self.cur = {"turn": turn, "t0": time.time()}
        self.cur.update(kw)

    def mark(self, stage, turn=None, once=True):
        if not self.cur or (turn is not None and turn != self.cur.get("turn")):
            return
        if once and stage in self.cur:
            return
        self.cur[stage] = time.time()

    def set(self, **kw):
        if self.cur:
            self.cur.update(kw)

    def flush(self):
        if self.cur and self.path:
            try:
                with open(self.path, "a") as f:
                    f.write(json.dumps(self.cur) + "\n")
            except OSError:
                pass
        self.cur = {}


def setup_logging(level=None):
    logging.basicConfig(level=level or os.environ.get("BENI_LOG", "INFO"),
                        format="%(asctime)s %(levelname).1s %(name)s: %(message)s", datefmt="%H:%M:%S")
