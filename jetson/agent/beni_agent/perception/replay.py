"""Sleep replay (§11.9 step 9, §15.1): at night on the charger, re-scan the day's recordings for what was missed live.

vision_core does the pixels ({op: replay}: NVDEC -> detector + face chain on the background stream, results on the
`replay` topic). This picks finished segments, names each face track against the gallery and clusters strangers
across nights (kv `replay_strangers`). A stranger seen on two different days becomes an 'unknown_visitor' episode
with a keyframe, and proactive asks about it ("Who was the person who visited yesterday afternoon?"). The keyframe is
flagged to the brain (`replay.flag`): Florence-2 captions it and looks for objects someone asked about and Beni
couldn't find (kv `wanted_objects`); `replay.result` adds that to the episode. Known people who appear only in the
recordings get a 'sighting' episode; wanted COCO objects the detector finds become object_sighting rows.
"""
import asyncio
import json
import logging
import os
import re
import time

import numpy as np

from beni_common import schemas as S
from beni_common.memory import new_id

from .identity import TAU, decide
from .thumbs import SIGHTING_GAP_S, _write
from ..util import spawn

log = logging.getLogger("replay")

STRIDE = 6                  # every 6th frame of 30 fps: 5 Hz, plenty for faces that stay a second
MIN_AGE_S = 120             # a segment still being written is younger than this
MAX_AGE_S = 36 * 3600       # only the last day and a half
SEG_TIMEOUT_S = 900
MIN_FACES = 3               # face embeddings per track before it counts
RECUR_DAYS = 2
MAX_STRANGERS = 200
STRANGER_KEEP_S = 7 * 86400   # §12.3: anonymous face clusters live at most 7 days unless named
MAX_DONE = 2000
IDLE_S = 30
SEG_NAME = re.compile(r"cam(\d)_(\d{8}-\d{6})\.ts$")


def seg_start(path):
    """Wall-clock start of a Recorder segment from its name (cam<N>_YYYYmmdd-HHMMSS.ts), else None."""
    m = SEG_NAME.search(path)
    if not m:
        return None
    return time.mktime(time.strptime(m.group(2), "%Y%m%d-%H%M%S"))


def unit(v):
    v = np.asarray(v, np.float32).ravel()
    n = float(np.linalg.norm(v))
    return v / n if n > 0 else v


def part_of_day(t):
    h = time.localtime(t).tm_hour
    return "morning" if 5 <= h < 12 else "afternoon" if h < 17 else "evening" if h < 22 else "night"


class Replay:
    def __init__(self, store, gallery, request, rec_dir, thumbs_root, link=None, clock=time.time):
        self.store, self.g, self.request, self.rec_dir, self.root = store, gallery, request, rec_dir, thumbs_root
        self.link, self.clock = link, clock
        self.mode = None
        self.cur = None             # path of the segment in flight
        self._done = None
        self._reset()

    def _reset(self):
        self.tracks = {}            # parent track id -> {"embs": [...], "jpeg": bytes|None, "pts": float}
        self.objs = {}              # wanted label -> (conf, pts)

    # ------------------------------------------------------------------ inputs
    def on_sched(self, topic, msg):
        self.mode = msg.get("mode")

    def on_msg(self, topic, msg):
        if topic != S.T_REPLAY or msg.get("path") != self.cur:
            return
        if msg.get("done"):
            if self._done is not None and not self._done.done():
                self._done.set_result(msg)
            return
        pts, jpeg, wanted = float(msg.get("pts") or 0.0), msg.get("jpeg"), self.wanted()
        for o in msg.get("objs", ()):
            if o.get("gie") == 2 and o.get("emb"):
                tr = self.tracks.setdefault(o.get("parent", -1), {"embs": [], "jpeg": None, "pts": pts})
                if len(tr["embs"]) < 30:
                    tr["embs"].append(np.frombuffer(o["emb"], np.float16).astype(np.float32))
                if jpeg and tr["jpeg"] is None:
                    tr["jpeg"] = jpeg
            elif o.get("gie", 1) == 1 and 0 < o.get("cls", 0) < len(S.COCO):
                label = S.COCO[o["cls"]]
                if label in wanted and o.get("conf", 0) > self.objs.get(label, (0.0, 0.0))[0]:
                    self.objs[label] = (float(o.get("conf", 0)), pts)

    def wanted(self):
        return set(self.store.kv_get("wanted_objects", []) or [])

    # ------------------------------------------------------------------ segments
    def pending(self):
        try:
            names = os.listdir(self.rec_dir)
        except OSError:
            return []
        done, now, out = set(self.store.kv_get("replay_done", []) or []), self.clock(), []
        for n in names:
            p = os.path.join(self.rec_dir, n)
            if not SEG_NAME.search(n) or n in done:
                continue
            try:
                age = now - os.stat(p).st_mtime
            except OSError:
                continue
            if MIN_AGE_S < age < MAX_AGE_S:
                out.append(p)
        return sorted(out, key=lambda p: seg_start(p) or 0)

    def _mark_done(self, path):
        done = self.store.kv_get("replay_done", []) or []
        done.append(os.path.basename(path))
        self.store.kv_set("replay_done", done[-MAX_DONE:])

    async def segment(self, path):
        """One segment through vision_core. -> True when finished (or unreadable), False when interrupted."""
        self._reset()
        self.cur, self._done = path, asyncio.get_running_loop().create_future()
        try:
            r = await self.request(S.EP["vision_ctrl"], {"op": "replay", "path": path, "stride": STRIDE})
            if not r.get("ok"):
                log.warning("replay %s refused: %s", path, r.get("err"))
                if r.get("err") in ("timeout", "replay off", "camera not running"):
                    return False
                self._mark_done(path)                   # "not a recording": never ask again
                return True
            t0 = self.clock()
            while not self._done.done():
                if self.mode != "replay" or self.clock() - t0 > SEG_TIMEOUT_S:
                    await self.request(S.EP["vision_ctrl"], {"op": "replay", "stop": 1})
                    return False
                await asyncio.wait([self._done], timeout=5)
            done = self._done.result()
            self.finish(path, done)
            self._mark_done(path)
            return True
        finally:
            self.cur = self._done = None

    # ------------------------------------------------------------------ results
    def finish(self, path, done=None):
        t0 = seg_start(path) or self.clock()
        cam = int(SEG_NAME.search(path).group(1)) if SEG_NAME.search(path) else 0
        seen, cl = {}, None
        for tid, tr in self.tracks.items():
            if len(tr["embs"]) < MIN_FACES:
                continue
            q = unit(np.mean(tr["embs"], 0))
            _, per, veto = self.g.scores(q)
            if veto:
                continue
            pid = decide(per)[0]
            t = t0 + tr["pts"]
            if pid is not None:
                seen.setdefault(pid, t)
            else:
                cl = self.store.kv_get("replay_strangers", []) or [] if cl is None else cl
                self.stranger(q, t, cam, tr["jpeg"], cl)
        if cl is not None:                              # one synced kv write per segment
            cl = sorted((c for c in cl if c["t"] > time.time() - STRANGER_KEEP_S), key=lambda c: -c["t"])
            self.store.kv_set("replay_strangers", cl[:MAX_STRANGERS])
        for pid, t in seen.items():
            self.known(pid, t)
        for label, (conf, pts) in self.objs.items():
            self.store.put("object_sighting", {"label": label, "cam": str(cam), "conf": conf, "t": t0 + pts,
                                               "attrs": json.dumps({"source": "replay"})})
        if self.objs:
            self.found({label: t0 + pts for label, (_, pts) in self.objs.items()})
        log.info("replayed %s: %d frames, %d face tracks, %d known, %d objects", os.path.basename(path),
                 (done or {}).get("frames", 0), len(self.tracks), len(seen), len(self.objs))
        self._reset()

    def known(self, pid, t):
        """A sighting episode unless live perception already logged one around that time."""
        like = '%"' + pid.replace('"', "") + '"%'
        if self.store.q("SELECT id FROM episode WHERE deleted=0 AND kind='sighting' AND people LIKE ? AND "
                        "t_start BETWEEN ? AND ?", (like, t - SIGHTING_GAP_S, t + SIGHTING_GAP_S)):
            return None
        p = self.store.get("person", pid) or {}
        text = "%s was here around %s (seen in the recordings)." % (p.get("display_name") or "Someone",
                                                                   time.strftime("%H:%M", time.localtime(t)))
        return self.store.add_episode(text, kind="sighting", people=[pid], importance=1.5, t_start=t, t_end=t)

    def stranger(self, q, t, cam, jpeg, cl):
        """Cluster an unknown face into `cl` across nights; returns the episode id when the cluster turns recurring."""
        day = time.strftime("%Y%m%d", time.localtime(t))
        best, bs = None, TAU
        for c in cl:
            s = float(np.frombuffer(bytes.fromhex(c["emb"]), np.float16).astype(np.float32) @ q)
            if s >= bs:
                best, bs = c, s
        if best is None:
            best = {"id": new_id(), "emb": "", "n": 0, "days": [], "t": t, "ep": None}
            cl.append(best)
        v = np.frombuffer(bytes.fromhex(best["emb"]), np.float16).astype(np.float32) if best["emb"] else q * 0
        best["emb"] = unit(v * best["n"] + q).astype(np.float16).tobytes().hex()
        best["n"] = min(best["n"] + 1, 50)
        if day not in best["days"]:
            best["days"] = (best["days"] + [day])[-14:]
        best["t"] = max(best["t"], t)
        ep = None
        if len(best["days"]) >= RECUR_DAYS and not best["ep"]:
            key = self._keyframe(jpeg, t)
            text = ("Someone I don't know was here in the %s of %s (also seen on %d other day%s)."
                    % (part_of_day(t), time.strftime("%A", time.localtime(t)), len(best["days"]) - 1,
                       "" if len(best["days"]) == 2 else "s"))
            ep = best["ep"] = self.store.add_episode(text, kind="unknown_visitor", importance=4.0, t_start=t,
                                                     t_end=t, keyframe_path=key)
            self.store.kv_set("ask_unknown", {"episode": ep, "t": t, "cluster": best["id"]})
            if jpeg:
                self.flag(ep, jpeg, cam, t)
        return ep

    def _keyframe(self, jpeg, t):
        if not jpeg:
            return None
        d = os.path.join(self.root, time.strftime("%Y%m%d", time.localtime(t)))
        p = os.path.join(d, "key-%s.jpg" % new_id())
        _write(d, p, jpeg)
        return p

    # ------------------------------------------------------------------ cloud (§11.9: Florence-2 on flagged frames)
    def flag(self, episode_id, jpeg, cam, t):
        if self.link is None or not self.link.online.is_set():
            return False
        spawn(self.link.send("replay.flag", id=episode_id, jpeg=jpeg, cam=cam, ts=t,
                                             want=sorted(self.wanted())))
        return True

    def on_result(self, f):
        """`replay.result{id, caption, found}` -> the caption on the episode, found objects as sightings."""
        ep = self.store.get("episode", f.get("id") or "")
        if ep is None:
            return
        cap = (f.get("caption") or "").strip()
        if cap:
            self.store.put("episode", {"id": ep["id"], "text": (ep["text"] + " Scene: " + cap)[:500]})
        for label in f.get("found") or ():
            self.store.put("object_sighting", {"label": label, "t": ep.get("t_start"), "thumb_path":
                                               ep.get("keyframe_path"), "attrs": json.dumps({"source": "replay"})})
        if f.get("found"):
            self.found({label: ep.get("t_start") or self.clock() for label in f["found"]})

    def found(self, seen):
        """{label: t} of wanted objects spotted in recordings: no longer wanted; the proactive loop mentions them
        (kv `found_objects`, lost-item hint §12.2)."""
        wanted = self.store.kv_get("wanted_objects", []) or []
        hits = [{"label": k, "t": t} for k, t in sorted(seen.items()) if k in wanted]
        if hits:
            self.store.kv_set("wanted_objects", [w for w in wanted if w not in seen])
            old = [f for f in self.store.kv_get("found_objects", []) or [] if f["label"] not in seen]
            self.store.kv_set("found_objects", (old + hits)[-10:])

    # ------------------------------------------------------------------ loop
    async def run(self):
        while True:
            await asyncio.sleep(IDLE_S)
            try:
                while self.mode == "replay":
                    todo = self.pending()
                    if not todo or not await self.segment(todo[0]):
                        break
            except Exception:
                log.exception("replay")
