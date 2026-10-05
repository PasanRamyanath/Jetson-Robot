"""memory.delta / memory.ack exchange (§11.5), identical on both peers (Python >= 3.8, asyncio).

Push: rows with hlc > cursor(peer) (own rows only; the peer's echoes are filtered), one batch in flight;
only the peer's ack advances the cursor, and a lost ack simply resends (apply_delta is idempotent).
"""
import asyncio
import logging
import time

from .store import SYNCED

log = logging.getLogger("sync")
PERIOD_S = 30.0


def max_hlc(store, node=None):
    """Newest hlc in the DB (optionally only rows written by `node`)."""
    best = ""
    for t in SYNCED:
        if node:
            h = store.q("SELECT max(hlc) AS h FROM %s WHERE hlc LIKE ?" % t, ("%-" + node,))[0]["h"]
        else:
            h = store.q("SELECT max(hlc) AS h FROM %s" % t)[0]["h"]
        best = max(best, h or "")
    return best


class DeltaSync:
    """send: async fn(type, **kw) -> bool.  online: fn() -> bool.  on_applied: fn(set_of_tables)."""

    def __init__(self, store, send, peer, peer_node, online=lambda: True, on_applied=None, batch=300):
        self.store, self.send, self.peer, self.peer_node = store, send, peer, peer_node
        self.online, self.on_applied, self.batch = online, on_applied, batch
        self._nudge = asyncio.Event()
        self._inflight = None
        self._sent_at = 0.0
        self.ready = False

    def nudge(self):
        self._nudge.set()

    def reset(self, cursor):
        """Session start: the peer holds everything of ours up to `cursor`."""
        self.store.set_peer_cursor(self.peer, cursor or "")
        self._inflight, self.ready = None, True
        self.nudge()

    async def on_frame(self, f):
        t = f.get("type")
        if t == "memory.delta":
            rows = f.get("rows") or {}
            applied, top = self.store.apply_delta(rows)
            await self.send("memory.ack", hlc=f.get("hlc") or top)
            if applied and self.on_applied:
                self.on_applied(set(rows))
            return applied
        if t == "memory.ack" and self._inflight is not None and f.get("hlc") == self._inflight:
            self.store.set_peer_cursor(self.peer, self._inflight)
            self._inflight = None
            self.nudge()                # a truncated batch may have more behind it
        return 0

    async def push_once(self):
        if not self.ready or self._inflight is not None or not self.online():
            return 0
        since = self.store.peer_cursor(self.peer)
        delta, cursor = self.store.changes_since(since, limit=self.batch, exclude_node=self.peer_node)
        if cursor == since:
            return 0
        if not delta:                   # only the peer's own rows: nothing to send, just advance
            self.store.set_peer_cursor(self.peer, cursor)
            return 0
        self._inflight, self._sent_at = cursor, time.monotonic()
        if not await self.send("memory.delta", rows=delta, hlc=cursor):
            self._inflight = None
            return 0
        return sum(len(v) for v in delta.values())

    async def run(self):
        while True:
            try:
                await asyncio.wait_for(self._nudge.wait(), PERIOD_S)
            except asyncio.TimeoutError:
                pass
            self._nudge.clear()
            if self._inflight is not None and time.monotonic() - self._sent_at >= PERIOD_S:
                self._inflight = None   # ack lost: resend (frequent nudges must not starve this)
            try:
                await self.push_once()
            except Exception:
                log.exception("sync push")
