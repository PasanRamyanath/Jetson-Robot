"""Hybrid retrieval (§11.6): dense (fp16 brute force) + FTS5 BM25, fused and scored Generative-Agents style.

Brute force beats an ANN index at this scale (<100k episodes): 20k x 384 fp16 = 15 MB, one pass ~5-10 ms on
the Nano's A57s when chunked, zero build time, exact results, trivially incremental.
"""
import json
import re
import threading
import time

import numpy as np

from .store import from_blob

_TOK = re.compile(r"\w+", re.UNICODE)
_CHUNK = 4096


def score(ep, q_sim, now, bm25_rank=None, w=(1.0, 1.0, 1.5, 0.5)):
    hours = max(0.0, (now - (ep.get("last_access") or ep.get("t_end") or now)) / 3600.0)
    recency = 0.995 ** hours                     # ~50% per 6 days without access
    importance = (ep.get("importance") or 3.0) / 10.0
    lexical = 0.0 if bm25_rank is None else 1.0 / (1 + bm25_rank)
    return w[0] * recency + w[1] * importance + w[2] * q_sim + w[3] * lexical


def fts_escape(query, max_terms=12):
    """User text -> safe FTS5 query: each token quoted, OR-joined (no syntax errors on quotes/colons/NEAR)."""
    toks = [t for t in _TOK.findall(query.lower()) if len(t) > 1][:max_terms]
    return " OR ".join('"%s"' % t for t in toks)


class VectorIndex(object):
    """Exact cosine search over L2-normalised fp16 vectors with amortised O(1) add/remove."""

    def __init__(self, dim=384, capacity=1024):
        self.dim = dim
        self.mat = np.zeros((capacity, dim), np.float16)
        self.ids = []
        self.pos = {}

    def __len__(self):
        return len(self.ids)

    def add(self, key, vec):
        v = np.asarray(vec, np.float16).ravel()
        if v.shape[0] != self.dim:
            return
        i = self.pos.get(key)
        if i is None:
            i = len(self.ids)
            if i == self.mat.shape[0]:
                self.mat = np.concatenate([self.mat, np.zeros_like(self.mat)])
            self.ids.append(key)
            self.pos[key] = i
        self.mat[i] = v

    def remove(self, key):
        i = self.pos.pop(key, None)
        if i is None:
            return
        last = len(self.ids) - 1              # swap-remove
        if i != last:
            self.mat[i] = self.mat[last]
            self.ids[i] = self.ids[last]
            self.pos[self.ids[i]] = i
        self.ids.pop()

    def search(self, q, k=48):
        n = len(self.ids)
        if n == 0:
            return [], np.zeros(0, np.float32)
        q = np.asarray(q, np.float32).ravel()
        sims = np.empty(n, np.float32)
        for s in range(0, n, _CHUNK):          # bounded fp32 temporaries
            sims[s:s + _CHUNK] = self.mat[s:min(n, s + _CHUNK)].astype(np.float32) @ q
        k = min(k, n)
        top = np.argpartition(-sims, k - 1)[:k]
        top = top[np.argsort(-sims[top])]
        return [self.ids[i] for i in top], sims[top]


class Retriever(object):
    """retrieve() for the prompt; context() packs person cards + facts + episodes into a char budget."""

    def __init__(self, store, embed, dim=384, reranker=None, nodes=("jetson", "kaggle")):
        self.store, self.embed, self.reranker = store, embed, reranker
        self.index = VectorIndex(dim)
        self._cursor = dict.fromkeys(set(nodes) | {store.node}, "")
        self._lock = threading.Lock()       # refresh() runs on the loop and in recall()'s worker thread
        self.refresh()

    def refresh(self):
        """Pull episode changes since the last refresh (local writes and applied sync deltas). One cursor per writer
        node: a peer's rows arrive late but in its own hlc order, so a single cursor would skip them."""
        n = 0
        with self._lock:
            for node, cur in self._cursor.items():
                rows = self.store.q("SELECT id, emb, deleted, hlc FROM episode WHERE hlc > ? AND hlc LIKE ? "
                                    "ORDER BY hlc", (cur, "%-" + node))
                for r in rows:
                    if r["deleted"] or not r["emb"]:
                        self.index.remove(r["id"])
                    else:
                        self.index.add(r["id"], from_blob(r["emb"]))
                    self._cursor[node] = r["hlc"]
                n += len(rows)
        return n

    def retrieve(self, query, speaker=None, k=8, now=None, touch=True):
        now = now or time.time()
        q = np.asarray(self.embed(query), np.float32).ravel()
        with self._lock:
            dense_ids, dense_sims = self.index.search(q, 48)
        sim = dict(zip(dense_ids, dense_sims.tolist()))
        bm25 = {}
        fq = fts_escape(query)
        if fq:
            for rank, r in enumerate(self.store.q(
                    "SELECT e.id FROM episode_fts f JOIN episode e ON e.rowid = f.rowid "
                    "WHERE episode_fts MATCH ? AND e.deleted = 0 ORDER BY f.rank LIMIT 24", (fq,))):
                bm25[r["id"]] = rank
        cand = list(set(sim) | set(bm25))
        if not cand:
            return []
        rows = self.store.q("SELECT id, t_start, t_end, kind, people, text, importance, emb, last_access "
                            "FROM episode WHERE deleted = 0 AND id IN (%s)" % ",".join("?" * len(cand)), cand)
        for r in rows:
            s = sim.get(r["id"])
            if s is None:                         # lexical-only hit: compute its dense sim directly
                e = from_blob(r["emb"])
                s = float(e.astype(np.float32) @ q) if e is not None and e.shape[0] == q.shape[0] else 0.0
            bonus = 0.3 if speaker and ('"%s"' % speaker) in (r["people"] or "") else 0.0
            r["_score"] = score(r, s, now, bm25.get(r["id"])) + bonus
            r.pop("emb", None)
        rows.sort(key=lambda r: r["_score"], reverse=True)
        if self.reranker is not None:             # Kaggle: bge-reranker over the top-24
            rows = self.reranker(query, rows[:24])
        rows = rows[:k]
        if touch:
            self.store.touch_access([r["id"] for r in rows], now)
        return rows

    def context(self, query, people=(), speaker=None, k=8, budget_chars=3600):
        """~900 tokens: rules (§11.11) and person cards always, then top facts per present person, then episodes."""
        out, used = {"rules": [], "cards": [], "facts": [], "episodes": []}, 0

        def fits(s):
            nonlocal used
            if used + len(s) > budget_chars:
                return False
            used += len(s)
            return True

        # Explicit rules: every active one, present people's first (a room rule matters most when its owner is away).
        rules = self.store.q("SELECT subject_id, text FROM fact WHERE deleted=0 AND status='active' AND "
                             "predicate='rule' ORDER BY recorded_at DESC LIMIT 40")
        rules.sort(key=lambda r: r["subject_id"] not in people)
        out["rules"] = [r["text"] for r in rules[:12]]           # outside the budget: rules are never dropped
        for c in self.store.person_cards(list(people)):
            if c.get("card") and fits(c["card"]):
                out["cards"].append({"id": c["id"], "name": c["display_name"], "card": c["card"]})
        for p in people:
            for f in self.store.facts_about(p, limit=6):
                if f["predicate"] != "rule" and fits(f["text"]):
                    out["facts"].append(f["text"])
        for e in self.retrieve(query, speaker=speaker, k=k):
            line = "[%s] %s" % (time.strftime("%Y-%m-%d %H:%M", time.localtime(e["t_end"] or 0)), e["text"])
            if not fits(line):
                break
            out["episodes"].append(line)
        return out


def format_context(ctx):
    """Render context() output as compact prompt text."""
    parts = []
    if ctx.get("rules"):
        parts.append("## House rules (always follow)\n" + "\n".join("- " + r for r in ctx["rules"]))
    for c in ctx["cards"]:
        parts.append("## %s\n%s" % (c["name"] or c["id"], c["card"]))
    if ctx["facts"]:
        parts.append("## Known facts\n" + "\n".join("- " + f for f in ctx["facts"]))
    if ctx["episodes"]:
        parts.append("## Relevant memories\n" + "\n".join("- " + e for e in ctx["episodes"]))
    return "\n\n".join(parts)


def people_of(row):
    try:
        return json.loads(row.get("people") or "[]")
    except ValueError:
        return []
