"""Jetson side of memory sync (§11.5): shared DeltaSync + session-start snapshot upload.

hello_ok carries `memory_since` = the newest Jetson-written HLC the brain already holds (None -> empty brain, so
we upload a full zlib snapshot on /bulk first).
"""
import asyncio
import logging
import os
import sqlite3
import tempfile
import zlib

from beni_common.memory.sync import DeltaSync, max_hlc

log = logging.getLogger("sync")


def snapshot_bytes(store, level=1):
    """Consistent online copy (sqlite backup API) -> zlib bytes. Run it in an executor."""
    db_file = store.q("PRAGMA database_list")[0]["file"]
    fd, tmp = tempfile.mkstemp(suffix=".db", dir=os.path.dirname(db_file) if db_file else None)
    os.close(fd)
    try:
        dst = sqlite3.connect(tmp)
        with store.lock:
            store.db.backup(dst)
        dst.close()
        with open(tmp, "rb") as f:
            return zlib.compress(f.read(), level)
    finally:
        os.unlink(tmp)


class MemorySync(DeltaSync):
    def __init__(self, store, link, on_applied=None):
        super().__init__(store, link.send, "brain", "kaggle", online=link.online.is_set, on_applied=on_applied)
        self.link = link

    async def on_hello(self, hello):
        self.ready, self._inflight = False, None
        since = hello.get("memory_since")
        if since is None:
            log.info("brain has no memory: uploading snapshot")
            top = max_hlc(self.store)
            data = await asyncio.get_running_loop().run_in_executor(None, snapshot_bytes, self.store)
            ack = await self.link.bulk_put("memory_snapshot", "memory.db.z", data, {"hlc": top, "codec": "zlib"})
            if not ack or not ack.get("ok"):
                log.warning("snapshot upload failed; falling back to a full delta sync")
                top = ""
            since = top
        self.reset(since)
