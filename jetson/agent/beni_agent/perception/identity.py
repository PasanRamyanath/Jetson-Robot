"""Open-set face/voice identification with prototype exemplars (§11.8). No training, O(N·d) per query.

Gallery: fp16 matrix of L2-normalised exemplars + person labels, loaded from face_exemplar/voice_exemplar.
Tracks:  per (cam, person-track-id) evidence is averaged over >= 3 frames before an identity is announced.
Learning: confident, *novel* embeddings are auto-added (cap 40/person, least-matched evicted); exemplars added this
session never justify further additions (drift guard); do_not_learn prototypes veto matches and learning.
"""
import time

import numpy as np

from beni_common.memory import from_blob, to_blob

TAU = 0.45           # accept (unaligned MobileFaceNet; tune per install)
MARGIN = 0.08
TAU_HIGH = TAU + 0.10
NOVEL_MAX_SIM = 0.75 # auto-add only if the best own exemplar is below this (new lighting/glasses/angle)
CAP = 40
MIN_FRAMES = 3
MIN_FACE_H = 80
TRACK_TTL = 3.0
DNL_TAU = 0.50       # similarity to a do-not-learn prototype that vetoes a track
TOUCH_GAP_S = 30.0   # match_count bookkeeping at most this often per exemplar (not one DB write per frame)


class Gallery:
    def __init__(self, store, table="face_exemplar", dim=128):
        self.store, self.table, self.dim = store, table, dim
        self.session_auto = set()
        self._touched = {}
        self.reload()

    def reload(self):
        rows = self.store.q("SELECT e.id, e.person_id, e.emb, e.source, e.match_count, e.created FROM %s e "
                            "JOIN person p ON p.id = e.person_id WHERE e.deleted = 0 AND p.deleted = 0"
                            % self.table)
        keep = [r for r in rows if r["emb"] and len(r["emb"]) // 2 == self.dim] if rows else []
        if rows and not keep:                       # first rows define the dimension (model swap safe)
            self.dim = len(rows[0]["emb"]) // 2
            keep = [r for r in rows if len(r["emb"]) // 2 == self.dim]
        pos = [r for r in keep if r["source"] != "do_not_learn"]
        neg = [r for r in keep if r["source"] == "do_not_learn"]
        self.ids = [r["id"] for r in pos]
        self.labels = np.array([r["person_id"] for r in pos], dtype=object)
        self.source = [r["source"] for r in pos]
        self.E = (np.stack([from_blob(r["emb"]) for r in pos]) if pos else np.zeros((0, self.dim), np.float16))
        self.N = (np.stack([from_blob(r["emb"]) for r in neg]) if neg else np.zeros((0, self.dim), np.float16))
        self.neg_labels = [r["person_id"] for r in neg]
        self.session_auto &= set(self.ids)          # drift guard survives evictions and sync reloads
        self.people = sorted(set(self.labels.tolist()))
        self._counts = {}
        for p in self.labels.tolist():
            self._counts[p] = self._counts.get(p, 0) + 1

    def scores(self, q):
        """-> (sims per exemplar fp32, {person: max sim}, veto: bool)."""
        q = np.asarray(q, np.float32).ravel()
        veto = bool(len(self.N)) and float((self.N.astype(np.float32) @ q).max()) >= DNL_TAU
        if not len(self.E):
            return np.zeros(0, np.float32), {}, veto
        s = self.E.astype(np.float32) @ q
        per = {}
        for i in np.argsort(-s)[:64]:               # people are few; top-64 exemplars cover them
            p = self.labels[i]
            if p not in per:
                per[p] = float(s[i])
        return s, per, veto

    def add(self, person_id, emb, source="auto", **extra):
        n = self._counts.get(person_id, 0)
        if n >= CAP:
            self._evict(person_id)
        row = {"person_id": person_id, "emb": to_blob(emb), "source": source, "created": time.time()}
        row.update(extra)
        eid = self.store.put(self.table, row)
        v = np.frombuffer(row["emb"], np.float16)[None, :]
        self.E = np.concatenate([self.E, v]) if len(self.E) else v.copy()
        self.ids.append(eid)
        self.labels = np.append(self.labels, np.array([person_id], dtype=object))
        self.source.append(source)
        self._counts[person_id] = self._counts.get(person_id, 0) + 1
        if source == "auto":
            self.session_auto.add(eid)
        if person_id not in self.people:
            self.people.append(person_id)
        return eid

    def _evict(self, person_id):
        """Replace the least useful auto exemplar (never enroll/confirmed ones)."""
        rows = self.store.q("SELECT id FROM %s WHERE person_id=? AND deleted=0 AND source='auto' "
                            "ORDER BY match_count ASC, created ASC LIMIT 1" % self.table, (person_id,))
        if rows:
            self.store.tombstone(self.table, rows[0]["id"])
            self.reload()

    def touch(self, exemplar_idx, now=None):
        eid, now = self.ids[exemplar_idx], now or time.time()
        if now - self._touched.get(eid, 0.0) < TOUCH_GAP_S:
            return
        self._touched[eid] = now
        with self.store.lock:                        # local bookkeeping; no hlc bump (like touch_access)
            self.store.db.execute("UPDATE %s SET match_count=match_count+1, last_matched=? WHERE id=?"
                                  % self.table, (now, eid))


def decide(per, tau=TAU, margin=MARGIN):
    """Top-1 person if it clears threshold and margin, else None. Returns (person|None, s1, s2)."""
    if not per:
        return None, 0.0, 0.0
    ranked = sorted(per.items(), key=lambda kv: -kv[1])
    s1 = ranked[0][1]
    s2 = ranked[1][1] if len(ranked) > 1 else 0.0
    return (ranked[0][0] if s1 >= tau and s1 - s2 >= margin else None), s1, s2


class _Track:
    __slots__ = ("key", "n", "acc", "person", "first", "last", "veto", "announced", "unknown_sent", "best_emb")

    def __init__(self, key, now):
        self.key, self.n, self.acc, self.person = key, 0, {}, None
        self.first = self.last = now
        self.veto, self.announced, self.unknown_sent, self.best_emb = False, False, False, None


class FaceIdentifier:
    """Feed det frames; emits ('person_seen', pid, key) / ('unknown_face', None, key) / ('person_left', pid, key)."""

    def __init__(self, gallery, learn=True, clock=time.time):
        self.g, self.learn, self.clock = gallery, learn, clock
        self.tracks = {}
        self.enroll_buf = None         # list of embeddings while enrolling
        self.enroll_face = None        # (cam, bbox) of the last face collected, for the thumbnail
        self.on_exemplar = None        # (eid, cam, bbox) -> None: a new exemplar wants its thumbnail (§3.17)

    def present(self):
        """People with a live track: stale tracks count as gone even when no frames arrive (privacy, cam stopped)."""
        now = self.clock()
        return sorted({t.person for t in self.tracks.values() if t.person and now - t.last <= TRACK_TTL})

    def on_frame(self, frame):
        now, out = self.clock(), []
        cam = frame.get("cam", 0)
        for o in frame.get("objs", ()):
            e = o.get("emb")
            if e is None or o.get("bbox", [0, 0, 0, 0])[3] * 720 / 288 < MIN_FACE_H:   # mux 288 px -> sensor 720 px
                continue
            q = np.frombuffer(e, np.float16).astype(np.float32)
            if q.shape[0] != self.g.dim and len(self.g.E):
                continue
            key = (cam, o.get("parent", -1) if o.get("parent", -1) >= 0 else o.get("tid"))
            t = self.tracks.get(key)
            if t is None:
                t = self.tracks[key] = _Track(key, now)
            t.last = now
            if self.enroll_buf is not None and len(self.enroll_buf) < 60:
                self.enroll_buf.append(q)
                self.enroll_face = (cam, o.get("bbox"))
            sims, per, veto = self.g.scores(q)
            t.veto = t.veto or veto
            t.n += 1
            for p, s in per.items():
                t.acc[p] = t.acc.get(p, 0.0) + s
            if t.veto:
                continue
            if t.person is None and t.n >= MIN_FRAMES:
                mean = {p: s / t.n for p, s in t.acc.items()}
                pid, s1, _ = decide(mean)
                if pid is not None:
                    t.person = pid
                elif not t.unknown_sent and now - t.first > 2.0:
                    t.unknown_sent = True
                    t.best_emb = q
                    out.append(("unknown_face", None, key))
            if t.person is not None:
                if not t.announced:
                    t.announced = True
                    out.append(("person_seen", t.person, key))
                if len(sims):
                    self._maybe_learn(t.person, q, sims, cam, o.get("bbox"))
        for key in [k for k, t in self.tracks.items() if now - t.last > TRACK_TTL]:
            t = self.tracks.pop(key)
            if t.announced:
                out.append(("person_left", t.person, key))
        return out

    def _maybe_learn(self, pid, q, sims, cam=0, bbox=None):
        mine = np.where(self.g.labels == pid)[0]
        if not len(mine):
            return
        i = mine[np.argmax(sims[mine])]
        best = float(sims[i])
        self.g.touch(int(i))
        if (self.learn and TAU_HIGH <= best < NOVEL_MAX_SIM and self.g.ids[i] not in self.g.session_auto):
            per = {}
            for j in np.argsort(-sims)[:8]:
                per.setdefault(self.g.labels[j], float(sims[j]))
            pid2, s1, s2 = decide(per, TAU_HIGH, MARGIN + 0.04)
            if pid2 == pid:
                self._thumb(self.g.add(pid, q, "auto"), cam, bbox)

    def _thumb(self, eid, cam, bbox):
        if self.on_exemplar is not None and bbox:
            self.on_exemplar(eid, cam, bbox)

    # ------------------------------------------------------------------ enrollment
    def start_enroll(self):
        self.enroll_buf = []

    def finish_enroll(self, person_id, k=12, source="enroll"):
        embs, self.enroll_buf = self.enroll_buf or [], None
        if len(embs) < 5:
            return 0
        face, self.enroll_face = self.enroll_face, None
        for i, v in enumerate(k_center(np.stack(embs), k)):
            eid = self.g.add(person_id, v, source)
            if i == 0 and face:
                self._thumb(eid, *face)
        for t in self.tracks.values():               # the enrolled person is whoever was unknown in view
            if t.person is None and not t.veto:
                t.person, t.announced = person_id, True
        return min(k, len(embs))

    def track_embedding(self, key):
        t = self.tracks.get(key)
        return t.best_emb if t else None


def k_center(X, k):
    """Greedy k-center (farthest-point) selection of diverse exemplars."""
    X = np.asarray(X, np.float32)
    if len(X) <= k:
        return list(X)
    chosen = [int(np.argmax(X @ X.mean(0)))]       # start near the centroid
    d = 1.0 - X @ X[chosen[0]]
    while len(chosen) < k:
        i = int(np.argmax(d))
        chosen.append(i)
        d = np.minimum(d, 1.0 - X @ X[i])
    return [X[i] for i in chosen]
