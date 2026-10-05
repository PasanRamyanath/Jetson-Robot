"""Hybrid logical clock (§11.5). Python 3.6-safe.

Format "{wall_ms:013d}-{counter:04d}-{node}" compares lexicographically in causal order.
"""
import threading
import time

_MAX_C = 9999


def fmt(wall_ms, counter, node):
    return "%013d-%04d-%s" % (wall_ms, counter, node)


def parse(h):
    wall, c, node = h.split("-", 2)
    return int(wall), int(c), node


class HLC(object):
    def __init__(self, node, clock=time.time):
        if "-" in node:
            raise ValueError("node id must not contain '-'")
        self.node = node
        self._clock = clock
        self._l = 0
        self._c = 0
        self._lock = threading.Lock()

    def _bump(self, l, c):
        if c > _MAX_C:           # counter overflow: borrow 1 ms from the future
            l, c = l + 1, 0
        self._l, self._c = l, c
        return fmt(l, c, self.node)

    def now(self):
        """Timestamp for a local event/write."""
        with self._lock:
            pt = int(self._clock() * 1000)
            if pt > self._l:
                return self._bump(pt, 0)
            return self._bump(self._l, self._c + 1)

    def update(self, remote):
        """Merge a timestamp received from a peer; returns the new local timestamp."""
        rl, rc, _ = parse(remote)
        with self._lock:
            pt = int(self._clock() * 1000)
            l = max(self._l, rl, pt)
            if l == self._l == rl:
                c = max(self._c, rc) + 1
            elif l == self._l:
                c = self._c + 1
            elif l == rl:
                c = rc + 1
            else:
                c = 0
            return self._bump(l, c)


def wall_ms_prefix(t_seconds):
    """HLC lower bound for a wall-clock time (for 'older than' queries)."""
    return "%013d" % int(t_seconds * 1000)
