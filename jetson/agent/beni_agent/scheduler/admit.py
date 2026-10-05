"""Background-tier admission (§3.17): rolling tegrastats windows -> which engines may take background work.

Pure logic (no I/O) so it is unit-testable; the scheduler main loop feeds it samples every ~1 s.
"""
import collections
import time


class Stats:
    def __init__(self, clock=time.time):
        self.clock = clock
        self.samples = collections.deque(maxlen=64)      # (t, row)
        self.ram_avail_mb = 4096
        self.batt_pct = None
        self.on_charger = False
        self.voice_active = False

    def add(self, row, t=None):
        self.samples.append((t or self.clock(), row))

    def _avg(self, key, window):
        now = self.clock()
        xs = [r[key] for t, r in self.samples if now - t <= window and r.get(key) is not None]
        return sum(xs) / len(xs) if xs else 0.0

    def _last(self, key):
        for _, r in reversed(self.samples):
            if r.get(key) is not None:
                return r[key]
        return 0.0

    @property
    def gr3d_avg_2s(self):
        return self._avg("gpu", 2.5)

    @property
    def gr3d_avg_10s(self):
        return self._avg("gpu", 10)

    @property
    def cpu_avg_10s(self):
        return self._avg("cpu_avg", 10)

    @property
    def nvdec(self):
        return self._last("nvdec")

    @property
    def temp_gpu(self):
        return self._last("temp_gpu")


ADMIT_IF = {
    "gpu": lambda s: s.gr3d_avg_10s < 60 and s.temp_gpu < 72,
    "nvdec": lambda s: s.nvdec < 50,
    "cpu": lambda s: s.cpu_avg_10s < 55,
    "ram": lambda s: s.ram_avail_mb > 600,
    "power": lambda s: s.on_charger or (s.batt_pct if s.batt_pct is not None else 100) > 40,
}


def pause_if(s):
    return s.gr3d_avg_2s > 85 or s.temp_gpu > 78 or s.ram_avail_mb < 400 or s.voice_active


def admit(s):
    """-> {'pause': bool, 'engines': {name: bool}}. A job runs only if not paused and all its engines are admitted."""
    paused = pause_if(s)
    return {"pause": paused, "engines": {k: (not paused) and f(s) for k, f in ADMIT_IF.items()}}


def job_allowed(decision, needs):
    return not decision["pause"] and all(decision["engines"].get(n, False) for n in needs)
