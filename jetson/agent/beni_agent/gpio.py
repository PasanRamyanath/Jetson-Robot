"""40-pin GPIO inputs (§4 item 52, §13.4): ESP32 e-stop/bump line on pin 13, optional mic-mute button.

Plain sysfs with poll(POLLPRI) in one daemon thread. Jetson.GPIO does the same thing with more threads and RAM.
beni-agent.service exports the pins as root (ExecStartPre); this module only reads `value`.
"""
import logging
import os
import select
import threading
import time

log = logging.getLogger("gpio")

SYSFS = "/sys/class/gpio"


def setup(n, edge="both", root=SYSFS):
    """Export and configure one pin if we are allowed to; returns the path of its `value` file (or None)."""
    d = os.path.join(root, "gpio%d" % n)
    try:
        if not os.path.isdir(d):
            with open(os.path.join(root, "export"), "w") as f:
                f.write(str(n))
            time.sleep(0.1)                      # udev chgrp race after export
        for name, v in (("direction", "in"), ("edge", edge)):
            try:
                with open(os.path.join(d, name)) as f:
                    if f.read().strip() == v:
                        continue
                with open(os.path.join(d, name), "w") as f:
                    f.write(v)
            except OSError:
                pass                             # set by ExecStartPre; value is what matters
        return os.path.join(d, "value") if os.access(os.path.join(d, "value"), os.R_OK) else None
    except OSError as e:
        log.warning("gpio%d unavailable: %s", n, e)
        return None


class Debounce:
    """Level changes shorter than `hold_s` are ignored (bounce); returns the new stable level or None."""

    def __init__(self, level, hold_s):
        self.level, self.hold_s = level, hold_s
        self.pending, self.since = None, 0.0

    def feed(self, level, t):
        if level == self.level:
            self.pending = None
            return None
        if self.pending != level:
            self.pending, self.since = level, t
        if t - self.since >= self.hold_s:
            self.level, self.pending = level, None
            return level
        return None


class Watcher:
    """Calls `on_change(name, level)` on the asyncio loop when a pin's debounced level changes."""

    def __init__(self, loop, on_change):
        self.loop, self.on_change = loop, on_change
        self.pins = {}                                   # fd -> (name, file, Debounce)
        self.poll = select.poll()

    def add(self, name, value_path, hold_s=0.0):
        if not value_path:
            return False
        f = open(value_path, "rb", buffering=0)
        level = self._read(f)
        self.pins[f.fileno()] = (name, f, Debounce(level, hold_s))
        self.poll.register(f.fileno(), select.POLLPRI | select.POLLERR)
        self.loop.call_soon_threadsafe(self.on_change, name, level)
        return True

    @staticmethod
    def _read(f):
        f.seek(0)
        return f.read(1) == b"1"

    def start(self):
        if self.pins:
            threading.Thread(target=self._run, name="gpio", daemon=True).start()
        return bool(self.pins)

    def _run(self):
        while True:
            # A 50 ms timeout re-samples pending bounces; with no pending edge this is one wake-up per second.
            pending = any(d.pending is not None for _, _, d in self.pins.values())
            events = self.poll.poll(50 if pending else 1000)
            now = time.monotonic()
            ready = {fd for fd, _ in events}
            for fd, (name, f, deb) in self.pins.items():
                if fd in ready or deb.pending is not None:
                    level = deb.feed(self._read(f), now)
                    if level is not None:
                        self.loop.call_soon_threadsafe(self.on_change, name, level)
