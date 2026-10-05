"""Brain-side mirror of the Jetson memory (§11.1, §11.5): same schema/store, node id "kaggle".

Session start: hello_ok.memory_since = newest Jetson-written HLC we hold, or None when the mirror is empty, in which
case the Jetson uploads a zlib snapshot on /bulk and load_snapshot() merges it with LWW (so rows we wrote meanwhile
survive). After that both sides exchange memory.delta frames (beni_common.memory.sync.DeltaSync).
"""
import logging
import os
import tempfile
import zlib

import numpy as np

from beni_common.memory import MemoryStore, Retriever, from_blob
from beni_common.memory.sync import DeltaSync, max_hlc

log = logging.getLogger("mirror")
NODE, PEER = "kaggle", "jetson"


class Mirror:
    def __init__(self, db, embed, reranker=None):
        self.store = MemoryStore(db, node=NODE)
        self.embed = embed
        self.retriever = Retriever(self.store, embed, reranker=reranker)
        self.loaded = bool(max_hlc(self.store, PEER))
        self.sync = None

    # ------------------------------------------------------------------ session / sync
    def memory_since(self):
        return max_hlc(self.store, PEER) if self.loaded else None

    def attach(self, send, online):
        """New /ws session: fresh DeltaSync bound to it. Starts pushing once the mirror holds the Jetson's data."""
        self.sync = DeltaSync(self.store, send, PEER, PEER, online=online, on_applied=self._on_applied)
        if self.loaded:
            self.sync.reset(self.store.peer_cursor(PEER))
        return self.sync

    async def on_frame(self, f):
        if self.sync is None:
            return
        n = await self.sync.on_frame(f)
        if f.get("type") == "memory.delta" and not self.loaded:
            self.loaded = True                          # snapshot upload failed: the Jetson fell back to deltas
            self.sync.reset(self.store.peer_cursor(PEER))
        return n

    def nudge(self):
        if self.sync is not None:
            self.sync.nudge()

    def _on_applied(self, tables):
        if "episode" in tables:
            self.retriever.refresh()

    def load_snapshot(self, data, meta):
        """Blocking (run in a thread): merge a zlib'd SQLite snapshot into the mirror. Returns rows applied."""
        mine = max_hlc(self.store, NODE)
        fd, tmp = tempfile.mkstemp(suffix=".db", dir=os.path.dirname(os.path.abspath(self.store_path())))
        with os.fdopen(fd, "wb") as f:
            f.write(zlib.decompress(data) if meta.get("codec", "zlib") == "zlib" else data)
        snap = MemoryStore(tmp, node="snapshot")
        n, since = 0, ""
        try:
            while True:
                delta, cur = snap.changes_since(since, limit=5000)
                if cur == since:
                    break
                n += self.store.apply_delta(delta)[0]
                since = cur
        finally:
            snap.close()
            for p in (tmp, tmp + "-wal", tmp + "-shm"):
                if os.path.exists(p):
                    os.unlink(p)
        if not mine:                                    # nothing of ours to send: the Jetson already has it all
            self.store.set_peer_cursor(PEER, max(since, meta.get("hlc") or ""))
        self.loaded = True
        self.retriever.refresh()
        log.info("snapshot merged: %d rows (cursor %s)", n, self.store.peer_cursor(PEER))
        return n

    def activate(self):
        """Event-loop side of load_snapshot (asyncio.Event isn't thread-safe): start pushing our rows."""
        if self.sync is not None:
            self.sync.reset(self.store.peer_cursor(PEER))

    def store_path(self):
        return self.store.q("PRAGMA database_list")[0]["file"] or os.path.join(tempfile.gettempdir(), "x.db")

    # ------------------------------------------------------------------ lookups used by tools / extraction
    def person_id(self, name):
        """Display name or alias -> person id (case-insensitive), or None."""
        if not name:
            return None
        n = name.strip().lower()
        for r in self.store.q("SELECT id, display_name, aliases FROM person WHERE deleted=0"):
            if (r["display_name"] or "").lower() == n or ('"%s"' % n) in (r["aliases"] or "").lower():
                return r["id"]
        return None

    def similar_facts(self, subject_id, emb, k=5, min_sim=0.5):
        rows = self.store.q("SELECT id, text, emb FROM fact WHERE subject_id=? AND deleted=0 AND status IN "
                            "('active','pending_confirm')", (subject_id,))
        if not rows or emb is None:
            return []
        q = np.asarray(emb, np.float32)
        out = []
        for r in rows:
            e = from_blob(r["emb"])
            s = float(e.astype(np.float32) @ q) if e is not None and e.shape[0] == q.shape[0] else 0.0
            if s >= min_sim:
                out.append((s, r))
        out.sort(key=lambda x: -x[0])
        return [dict(r, sim=s) for s, r in out[:k]]
