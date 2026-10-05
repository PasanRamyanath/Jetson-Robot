"""MemoryStore: the single SQLite system of record (§11.4) with HLC last-writer-wins sync (§11.5).

Runs on SQLite 3.22 (Ubuntu 18.04): no UPSERT, so remote rows are applied with an explicit LWW check and
INSERT OR REPLACE. `recursive_triggers=ON` makes REPLACE fire the FTS delete trigger, keeping episode_fts exact.
Rows in synced tables are never hard-deleted except by purge_tombstones() (Jetson only, after 30 days).
"""
import json
import os
import sqlite3
import threading
import time
import uuid

import numpy as np

from ..hlc import HLC, wall_ms_prefix

SCHEMA = os.path.join(os.path.dirname(__file__), "schema.sql")

# table -> primary key; everything here is exchanged in memory.delta frames.
SYNCED = {
    "person": "id", "face_exemplar": "id", "voice_exemplar": "id", "episode": "id", "fact": "id",
    "place": "id", "object_sighting": "id", "object_belief": "id", "skill": "id", "routine": "id",
    "feedback": "id", "nav_experience": "id", "kv": "k",
}
# Columns cleared when a row is tombstoned (privacy: the content must not survive in the DB or on the peer).
_SCRUB = {
    "episode": {"text": "", "emb": None, "keyframe_path": None, "media_ref": None, "people": "[]"},
    "fact": {"text": "", "object": "", "emb": None},
    "face_exemplar": {"emb": b"", "thumb_path": None},
    "voice_exemplar": {"emb": b""},
    "person": {"card": None, "aliases": None},
    "feedback": {"prompt": None, "response": None, "better_response": None},
    "object_sighting": {"thumb_path": None, "emb": None},
}


def new_id(prefix=""):
    return prefix + uuid.uuid4().hex[:16]


def to_blob(vec):
    """fp16 bytes of an L2-normalised vector (384-d bge-small -> 768 bytes)."""
    v = np.asarray(vec, dtype=np.float32).ravel()
    n = float(np.linalg.norm(v))
    return (v / n if n > 0 else v).astype(np.float16).tobytes()


def from_blob(b):
    return np.frombuffer(b, dtype=np.float16) if b else None


class MemoryStore(object):
    def __init__(self, path, node="jetson", hlc=None):
        self.hlc = hlc or HLC(node)
        self.node = self.hlc.node
        if path != ":memory:":
            os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        self.db = sqlite3.connect(path, check_same_thread=False, isolation_level=None, timeout=10)
        self.db.row_factory = sqlite3.Row
        self.lock = threading.RLock()
        with open(SCHEMA) as f:
            self.db.executescript(f.read())
        for p in ("synchronous=NORMAL", "foreign_keys=ON", "recursive_triggers=ON", "temp_store=MEMORY",
                  "cache_size=-8000"):                     # 8 MB page cache: the Jetson is RAM-bound
            self.db.execute("PRAGMA " + p)
        self.cols = {t: [r[1] for r in self.db.execute("PRAGMA table_info(%s)" % t)] for t in SYNCED}
        top = max(self.db.execute("SELECT max(hlc) FROM %s" % t).fetchone()[0] or "" for t in SYNCED)
        if top:     # no RTC on the Nano: a boot clock behind the DB must not mint older hlcs (lost LWW, skipped sync)
            self.hlc.update(top)

    # ---------------------------------------------------------------- plumbing
    def tx(self):
        return _Tx(self)

    def q(self, sql, args=()):
        with self.lock:
            return [dict(r) for r in self.db.execute(sql, args)]

    def close(self):
        with self.lock:
            self.db.close()

    def _replace(self, table, row):
        cols = [c for c in self.cols[table] if c in row]
        self.db.execute("INSERT OR REPLACE INTO %s (%s) VALUES (%s)"
                        % (table, ",".join(cols), ",".join("?" * len(cols))),
                        [row[c] for c in cols])

    # ---------------------------------------------------------------- local writes
    def get(self, table, key, include_deleted=False):
        sql = "SELECT * FROM %s WHERE %s=?" % (table, SYNCED[table])
        if not include_deleted:
            sql += " AND deleted=0"
        rows = self.q(sql, (key,))
        return rows[0] if rows else None

    def put(self, table, row):
        """Insert or merge-update a row (partial dicts keep other columns). Returns the key."""
        pk = SYNCED[table]
        with self.tx():
            key = row.get(pk) or new_id()
            cur = self.get(table, key, include_deleted=True) or {}
            cur.update(row)
            cur[pk] = key
            cur["hlc"] = self.hlc.now()
            cur.setdefault("deleted", 0)
            self._replace(table, cur)
        return key

    def tombstone(self, table, key):
        """Soft delete + scrub content; the tombstone syncs so the peer deletes too."""
        with self.tx():
            cur = self.get(table, key, include_deleted=True)
            if cur is None or cur["deleted"]:
                return False
            cur.update(_SCRUB.get(table, {}))
            cur["deleted"], cur["hlc"] = 1, self.hlc.now()
            self._replace(table, cur)
        return True

    def kv_get(self, k, default=None):
        r = self.get("kv", k)
        return json.loads(r["v"]) if r else default

    def kv_set(self, k, v):
        self.put("kv", {"k": k, "v": json.dumps(v)})

    # ---------------------------------------------------------------- sync (§11.5)
    def changes_since(self, since="", limit=500, exclude_node=None):
        """Rows with hlc > since, at most ~limit per table. Returns ({table: [rows]}, cursor).

        If any table was truncated the cursor is the smallest last-hlc among truncated tables and rows beyond it
        are dropped, so repeated calls with the returned cursor never skip a row.
        """
        out, cut = {}, None
        with self.lock:
            for t in SYNCED:
                rows = [dict(r) for r in self.db.execute(
                    "SELECT * FROM %s WHERE hlc > ? ORDER BY hlc LIMIT ?" % t, (since, limit))]
                if len(rows) == limit:
                    cut = rows[-1]["hlc"] if cut is None else min(cut, rows[-1]["hlc"])
                out[t] = rows
        cursor = since
        for t in list(out):
            rows = [r for r in out[t] if cut is None or r["hlc"] <= cut]
            if rows:
                cursor = max(cursor, rows[-1]["hlc"])
            if exclude_node:                      # don't echo the peer's own rows back to it
                suffix = "-" + exclude_node
                rows = [r for r in rows if not r["hlc"].endswith(suffix)]
            if rows:
                out[t] = rows
            else:
                del out[t]
        return out, (cut or cursor)

    def apply_delta(self, delta):
        """Apply {table: [rows]} from a peer with LWW by hlc. Returns (applied, max_hlc_seen)."""
        applied, top = 0, ""
        with self.tx():
            for t, rows in delta.items():
                if t not in SYNCED:
                    continue
                pk = SYNCED[t]
                for row in rows:
                    h = row.get("hlc")
                    if not h or pk not in row:
                        continue
                    top = max(top, h)
                    cur = self.db.execute("SELECT hlc FROM %s WHERE %s=?" % (t, pk), (row[pk],)).fetchone()
                    if cur is None or h > cur[0]:
                        self._replace(t, row)
                        applied += 1
            if top:
                self.hlc.update(top)
        return applied, top

    def peer_cursor(self, peer):
        r = self.q("SELECT last_hlc FROM sync_state WHERE peer=?", (peer,))
        return r[0]["last_hlc"] if r else ""

    def set_peer_cursor(self, peer, h):
        with self.lock:
            self.db.execute("INSERT OR REPLACE INTO sync_state VALUES (?,?,?)", (peer, h, time.time()))

    def purge_tombstones(self, older_than_s=30 * 86400, now=None):
        """Hard-delete tombstones older than the window (Jetson only; peers have long since seen them)."""
        cutoff = wall_ms_prefix((now or time.time()) - older_than_s)
        n = 0
        with self.tx():
            for t in SYNCED:
                n += self.db.execute("DELETE FROM %s WHERE deleted=1 AND hlc < ?" % t, (cutoff,)).rowcount
        return n

    # ---------------------------------------------------------------- domain helpers
    def add_episode(self, text, kind="conversation", people=(), importance=3.0, emb=None, place_id=None,
                    t_start=None, t_end=None, **extra):
        now = time.time()
        row = {"text": text[:500], "kind": kind, "people": json.dumps(list(people)), "importance": importance,
               "emb": to_blob(emb) if emb is not None else None, "place_id": place_id,
               "t_start": t_start or now, "t_end": t_end or now}
        row.update(extra)
        return self.put("episode", row)

    def add_fact(self, subject_id, predicate, obj, text, emb=None, confidence=0.7, source_kind="said_by_user",
                 **extra):
        now = time.time()
        row = {"subject_id": subject_id, "predicate": predicate, "object": obj, "text": text,
               "emb": to_blob(emb) if emb is not None else None, "confidence": confidence,
               "status": "active", "valid_from": now, "recorded_at": now, "source_kind": source_kind}
        row.update(extra)
        return self.put("fact", row)

    def supersede_fact(self, old_id, **new):
        """UPDATE semantics (§11.7): new row + old row status='superseded' (bitemporal history kept)."""
        old = self.get("fact", old_id)
        if old is None:
            return None
        row = {k: v for k, v in old.items() if k not in ("id", "hlc", "deleted")}
        row.update(new)
        row["recorded_at"] = row["valid_from"] = time.time()
        with self.tx():
            nid = self.put("fact", row)
            self.put("fact", {"id": old_id, "status": "superseded", "superseded_by": nid, "valid_to": time.time()})
        return nid

    def touch_access(self, ids, now=None):
        """Retrieval refreshes recency. Local-only bookkeeping: it does not bump hlc (no sync churn)."""
        if not ids:
            return
        with self.lock:
            self.db.executemany("UPDATE episode SET access_count=access_count+1, last_access=? WHERE id=?",
                                [(now or time.time(), i) for i in ids])

    def person_cards(self, ids):
        if not ids:
            return []
        return self.q("SELECT id, display_name, relation, card FROM person WHERE deleted=0 AND id IN (%s)"
                      % ",".join("?" * len(ids)), list(ids))

    def facts_about(self, subject_id, limit=8):
        return self.q("SELECT id, predicate, object, text, confidence FROM fact WHERE subject_id=? AND deleted=0 "
                      "AND status='active' AND (valid_to IS NULL OR valid_to > ?) ORDER BY confidence DESC, "
                      "recorded_at DESC LIMIT ?", (subject_id, time.time(), limit))

    def forget_person(self, pid):
        """"Forget me" (§11.8.7): tombstone exemplars, facts and episodes; keep a do-not-learn negative prototype."""
        faces = self.q("SELECT id, emb FROM face_exemplar WHERE person_id=? AND deleted=0", (pid,))
        embs = [from_blob(f["emb"]).astype(np.float32) for f in faces if f["emb"]]
        n = 0
        with self.tx():
            for t, where in (("face_exemplar", "person_id=?"), ("voice_exemplar", "person_id=?"),
                             ("fact", "subject_id=? OR source_person=?"), ("episode", "people LIKE ?"),
                             ("routine", "person_id=?"), ("feedback", "person_id=?")):
                args = ('%%"%s"%%' % pid,) if t == "episode" else (pid,) * where.count("?")
                for r in self.q("SELECT id FROM %s WHERE deleted=0 AND (%s)" % (t, where), args):
                    n += self.tombstone(t, r["id"])
            if self.get("kv", "card_prev." + pid):             # previous core card (consolidation step 4)
                self.put("kv", {"k": "card_prev." + pid, "v": "null", "deleted": 1})
            self.put("person", {"id": pid, "do_not_learn": 1, "consent_face": 0, "consent_voice": 0,
                                "card": None, "aliases": None, "display_name": None})
            if embs:
                self.put("face_exemplar", {"person_id": pid, "emb": to_blob(np.mean(embs, axis=0)),
                                           "source": "do_not_learn", "created": time.time()})
        return n


class _Tx(object):
    """Re-entrant BEGIN IMMEDIATE / COMMIT around the store lock (nested use joins the outer transaction)."""

    def __init__(self, store):
        self.s = store

    def __enter__(self):
        self.s.lock.acquire()
        self.outer = not self.s.db.in_transaction
        if self.outer:
            self.s.db.execute("BEGIN IMMEDIATE")
        return self.s

    def __exit__(self, et, ev, tb):
        try:
            if self.outer:
                self.s.db.execute("ROLLBACK" if et else "COMMIT")
        finally:
            self.s.lock.release()
