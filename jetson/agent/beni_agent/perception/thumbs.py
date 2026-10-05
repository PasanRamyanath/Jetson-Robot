"""Memory thumbnails (§3.17 NVJPG row, §11.2 episodic keyframes): face-exemplar crops and sighting keyframes.

vision_core cuts them ({op: thumb}: VIC crop + scale, NVJPG); this only writes the JPEG next to memory.db (inside the
vault when there is one) and stores the path on the row. Files no live row points at (tombstones, "forget me") and
the oldest keyframes beyond the disk budget are swept hourly. On the DeepStream path the op is unknown: no thumbnails.
"""
import asyncio
import logging
import os
import time

from beni_common.memory import new_id

log = logging.getLogger("thumbs")

FACE_N, KEY_N = 128, 320     # 128x128 face (~5 KB), 320x180 keyframe (~12 KB)
FACE_PAD = 1.4               # the detector's face box is tight: keep hair and chin
SIGHTING_GAP_S = 900         # one sighting episode per person per 15 min
SWEEP_S, ORPHAN_AGE_S = 3600, 3600
DET_W, DET_H = 512.0, 288.0
REFS = (("face_exemplar", "thumb_path"), ("episode", "keyframe_path"), ("object_sighting", "thumb_path"))


def face_box(bbox, pad=FACE_PAD):
    """[l, t, w, h] detector px -> padded normalised {l, t, r, b} for {op: thumb}."""
    l, t, w, h = bbox
    cx, cy, hw, hh = (l + w / 2) / DET_W, (t + h / 2) / DET_H, w * pad / 2 / DET_W, h * pad / 2 / DET_H
    return {"l": max(0.0, cx - hw), "t": max(0.0, cy - hh), "r": min(1.0, cx + hw), "b": min(1.0, cy + hh)}


class Thumbs:
    def __init__(self, store, grab, root, keep_mb=256, clock=time.time):
        self.store, self.grab, self.root, self.keep, self.clock = store, grab, root, keep_mb << 20, clock
        self._seen = {}

    async def face(self, eid, cam, bbox):
        path = await self._save("face", cam, FACE_N, face_box(bbox))
        if path:
            self.store.put("face_exemplar", {"id": eid, "thumb_path": path})
        return path

    async def sighting(self, pid, name, cam=0, place=None):
        """An arrival -> a 'sighting' episode with a keyframe (rate-limited per person)."""
        now = self.clock()
        if now - self._seen.get(pid, 0.0) < SIGHTING_GAP_S:
            return None
        self._seen[pid] = now
        path = await self._save("key", cam, KEY_N, {})
        text = "Saw %s%s." % (name, " in the %s" % place if place else "")
        return self.store.add_episode(text, kind="sighting", people=[pid], importance=2.0, keyframe_path=path)

    async def _save(self, kind, cam, n, box):
        m = await self.grab(cam, "thumb", n=n, **box)
        if not m or not m.get("jpeg"):
            return None
        d = os.path.join(self.root, time.strftime("%Y%m%d", time.localtime(self.clock())))
        path = os.path.join(d, "%s-%s.jpg" % (kind, new_id()))
        await asyncio.get_running_loop().run_in_executor(None, _write, d, path, m["jpeg"])
        return path

    def referenced(self):
        out = set()
        for table, col in REFS:
            for r in self.store.q("SELECT %s AS p FROM %s WHERE deleted=0 AND %s IS NOT NULL" % (col, table, col)):
                out.add(r["p"])
        return out

    def sweep_files(self, keep):
        """Thread-safe (no DB): drop orphans older than an hour, then the oldest keyframes over budget."""
        now, files, removed = time.time(), [], 0
        for dp, _, names in os.walk(self.root):
            for n in names:
                p = os.path.join(dp, n)
                try:
                    st = os.stat(p)
                except OSError:
                    continue
                if p not in keep and now - st.st_mtime > ORPHAN_AGE_S:
                    removed += _rm(p)
                else:
                    files.append((st.st_mtime, st.st_size, p))
        total = sum(f[1] for f in files)
        for _, size, p in sorted(f for f in files if os.path.basename(f[2]).startswith("key-")):
            if total <= self.keep:
                break
            total -= size
            removed += _rm(p)
        for dp, dirs, names in os.walk(self.root, topdown=False):
            if dp != self.root and not dirs and not names:
                try:
                    os.rmdir(dp)
                except OSError:             # a thumbnail landed meanwhile
                    pass
        return removed

    async def run(self):
        while True:
            await asyncio.sleep(SWEEP_S)
            try:
                n = await asyncio.get_running_loop().run_in_executor(None, self.sweep_files, self.referenced())
                if n:
                    log.info("swept %d thumbnails", n)
            except Exception:
                log.exception("thumbnail sweep")


def _write(d, path, data):
    os.makedirs(d, exist_ok=True)
    with open(path, "wb") as f:
        f.write(data)


def _rm(p):
    try:
        os.remove(p)
        return 1
    except OSError:
        return 0
