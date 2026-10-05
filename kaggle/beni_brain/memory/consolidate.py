"""'Sleep' consolidation (§11.9). Runs once per session when the robot is idle, and on brain.stop before the kernel
exits. Every change is an ordinary store write, so it reaches the Jetson as memory.delta.

Steps here: 1 backlog, 2 day/week summaries + raw drop, 3 reflection, 4 cards, 5 fact hygiene, 7 forgetting, 8 object
beliefs, 10 routines, 11 skills, 12 exemplar outliers. Step 9 (sleep replay) is the Jetson's replay.py + the
replay.flag tool; step 13 is training/; step 6 (importance re-scoring) is not done: extraction already scores each
turn with its context.
"""
import asyncio
import json
import logging
import time

import numpy as np

from beni_common.hlc import wall_ms_prefix
from beni_common.memory import from_blob, routines, skills

from .. import prompts as P

log = logging.getLogger("consolidate")
DAY = 86400.0
FINE = ("conversation", "observation", "sighting", "unknown_visitor")   # what day summaries cover
STABLE = ("birthday", "born", "name", "relation", "rule", "anniversary", "allergy")   # never decayed (step 5c)
REFLECT_AT = 40.0           # summed importance since the last reflection (step 3)


def expire_facts(store, now):
    """Step 5a: facts past valid_to leave the active set."""
    rows = store.q("SELECT id FROM fact WHERE deleted=0 AND status='active' AND valid_to IS NOT NULL AND valid_to < ?",
                   (now,))
    for r in rows:
        store.put("fact", {"id": r["id"], "status": "expired"})
    return len(rows)


def dedupe_facts(store, sim=0.92):
    """Step 5b: near-duplicates (same subject + predicate, cosine > sim) -> keep the most confident."""
    n = 0
    groups = store.q("SELECT subject_id, predicate FROM fact WHERE deleted=0 AND status='active' AND emb IS NOT NULL "
                     "GROUP BY subject_id, predicate HAVING count(*) > 1")
    for g in groups:
        rows = store.q("SELECT id, emb, confidence FROM fact WHERE deleted=0 AND status='active' AND emb IS NOT NULL "
                       "AND subject_id=? AND predicate=? ORDER BY confidence DESC, recorded_at DESC",
                       (g["subject_id"], g["predicate"]))
        vs = [from_blob(r["emb"]).astype(np.float32) for r in rows]
        if len({v.shape[0] for v in vs}) > 1:           # embedder changed: not comparable until re-embedded
            continue
        E = np.stack(vs)
        S = E @ E.T
        dead = set()
        for i in range(len(rows)):
            if i in dead:
                continue
            for j in range(i + 1, len(rows)):
                if j not in dead and S[i, j] > sim:
                    dead.add(j)
                    store.put("fact", {"id": rows[j]["id"], "status": "superseded", "superseded_by": rows[i]["id"]})
                    n += 1
    return n


def decay_confidence(store, now, age_days=90, step=0.1, floor=0.3):
    """Step 5c: facts not written (added, re-confirmed or decayed) for 90 days lose confidence, stable ones excepted.
    The decay itself is a write, so a fact drops one step per 90 days until the floor."""
    rows = store.q("SELECT id, predicate, confidence FROM fact WHERE deleted=0 AND status='active' AND confidence > ? "
                   "AND hlc < ?", (floor + 1e-6, wall_ms_prefix(now - age_days * DAY)))
    n = 0
    for r in rows:
        if not any(s in (r["predicate"] or "") for s in STABLE):
            store.put("fact", {"id": r["id"], "confidence": round(max(floor, r["confidence"] - step), 3)})
            n += 1
    return n


def _day0(t):
    lt = time.localtime(t)
    return time.mktime((lt.tm_year, lt.tm_mon, lt.tm_mday, 0, 0, 0, 0, 0, -1))


def _summarised(store, t0, t1):
    return bool(store.q("SELECT 1 FROM episode WHERE deleted=0 AND kind='summary' AND t_start=? AND t_end=? LIMIT 1",
                        (t0, t1)))


async def _summary(store, llm, embed, span, when, rows, t0, t1):
    items = "\n".join("- %s%s" % (time.strftime("%H:%M ", time.localtime(r["t_start"])) if span == "day" else "",
                                   (r["text"] or "")[:300]) for r in rows)
    text = (await llm.complete(P.SUMMARY.format(span=span, when=when, items=items), max_tokens=220) or "").strip()
    if not text:
        return None
    people = sorted({p for r in rows for p in json.loads(r["people"] or "[]")})
    emb = await asyncio.to_thread(embed.doc, text) if embed is not None else None
    return store.add_episode(text, kind="summary", people=people, emb=emb, t_start=t0, t_end=t1,
                             importance=max(r["importance"] or 3 for r in rows),
                             parent_ids=json.dumps([r["id"] for r in rows]))


async def summarise(store, llm, embed, now, max_new=7, min_rows=3):
    """Step 2: finished days -> day summaries, finished Monday-to-Sunday weeks -> week summaries (evidence in
    parent_ids). Episodes are already per-turn summaries, so the hour level is skipped. Oldest first, `max_new` per
    pass; days with fewer than `min_rows` episodes keep their raw episodes and get no summary."""
    today, n = _day0(now), 0
    rows = store.q("SELECT id, kind, text, people, importance, t_start FROM episode WHERE deleted=0 AND kind IN "
                   "(%s) AND t_start >= ? AND t_start < ? ORDER BY t_start" % ",".join("?" * len(FINE)),
                   FINE + (today - 35 * DAY, today))
    days = {}
    for r in rows:
        days.setdefault(_day0(r["t_start"]), []).append(r)
    for d0 in sorted(days):
        d1 = _day0(d0 + 1.5 * DAY)
        if n >= max_new or len(days[d0]) < min_rows or _summarised(store, d0, d1):
            continue
        top = sorted(days[d0], key=lambda r: -(r["importance"] or 0))[:60]
        if await _summary(store, llm, embed, "day", time.strftime("%A %d %B %Y", time.localtime(d0)),
                          sorted(top, key=lambda r: r["t_start"]), d0, d1):
            n += 1
    week = _day0(today - time.localtime(today).tm_wday * DAY + 0.5 * DAY)     # this Monday 00:00
    for k in range(4, 0, -1):
        w0, w1 = _day0(week - 7 * k * DAY + 0.5 * DAY), _day0(week - 7 * (k - 1) * DAY + 0.5 * DAY)
        if n >= max_new or _summarised(store, w0, w1):
            continue
        ds = store.q("SELECT id, text, people, importance, t_start FROM episode WHERE deleted=0 AND kind='summary' "
                     "AND t_start >= ? AND t_end <= ? AND t_end - t_start < ? ORDER BY t_start", (w0, w1, 2 * DAY))
        if len(ds) >= 2 and await _summary(store, llm, embed, "week", "week of " + time.strftime(
                "%d %B %Y", time.localtime(w0)), ds, w0, w1):
            n += 1
    return n


def drop_raw(store, now, keep_days=30):
    """Step 2 (compression): after 30 days, episodes covered by a day summary and extracted offline transcripts
    are tombstoned; the summaries (and the facts) stay."""
    covered = set()
    for r in store.q("SELECT parent_ids FROM episode WHERE deleted=0 AND kind='summary' AND t_end - t_start < ?",
                     (2 * DAY,)):
        covered.update(json.loads(r["parent_ids"] or "[]"))
    rows = store.q("SELECT id, kind FROM episode WHERE deleted=0 AND t_end < ? AND kind IN (%s)"
                   % ",".join("?" * (len(FINE) + 1)), (now - keep_days * DAY,) + FINE + ("conversation_extracted",))
    n = 0
    for r in rows:
        if (r["kind"] == "conversation_extracted" or r["id"] in covered) and store.tombstone("episode", r["id"]):
            n += 1
    return n


async def reflect(mirror, llm, now, threshold=REFLECT_AT, max_people=5):
    """Step 3 (Generative Agents): once a person's summed episode importance since the last reflection passes the
    threshold, infer up to 3 insights -> reflection episodes (evidence in parent_ids) + candidate facts (<= 0.6)."""
    store, embed = mirror.store, mirror.embed
    last = store.kv_get("reflect.last", {}) or {}
    n = 0
    for p in store.q("SELECT id, display_name FROM person WHERE deleted=0 AND do_not_learn=0"):
        if n >= max_people:
            break
        pid = p["id"]
        rows = store.q("SELECT id, text, importance FROM episode WHERE deleted=0 AND kind IN ('conversation', "
                       "'observation', 'summary') AND t_end > ? AND people LIKE ? ORDER BY importance DESC, t_end DESC "
                       "LIMIT 30", (last.get(pid, 0), '%"' + pid + '"%'))
        if sum(r["importance"] or 0 for r in rows) < threshold:
            continue
        items = "\n".join("%d. %s" % (i + 1, (r["text"] or "")[:300]) for i, r in enumerate(rows))
        out = await llm.json(P.REFLECT.format(name=p["display_name"] or pid, items=items), P.REFLECT_SCHEMA,
                             max_tokens=500)
        for ins in (out.get("insights") or [])[:3]:
            text = (ins.get("text") or "").strip()
            if not text:
                continue
            ev = [rows[i - 1]["id"] for i in ins.get("evidence") or () if isinstance(i, int) and 0 < i <= len(rows)]
            emb = await asyncio.to_thread(embed.doc, text)
            ep = store.add_episode(text, kind="reflection", people=[pid], importance=6.0, emb=emb,
                                   parent_ids=json.dumps(ev))
            pred = (ins.get("predicate") or "").strip().lower().replace(" ", "_")[:40]
            if pred and ins.get("object") and not mirror.similar_facts(pid, emb, k=1, min_sim=0.85):
                sensitive = any(s in pred for s in P.SENSITIVE)
                store.add_fact(pid, pred, str(ins["object"])[:200], text[:300], emb=emb, confidence=0.5,
                               source_kind="inferred", source_episode=ep,
                               status="pending_confirm" if sensitive else "active")
        last[pid] = now
        n += 1
    if n:
        store.kv_set("reflect.last", last)
    return n


def prune_exemplars(store, z=2.5, floor=0.35, min_trusted=3):
    """Step 12 (§11.8 drift guard): per person and modality, auto-added exemplars far from the centroid of the
    trusted (enrolled/confirmed) ones -- cosine < max(floor, median - z*std of the trusted) -- are possible
    contamination and are tombstoned. Trusted exemplars are never removed here."""
    n = 0
    for table in ("face_exemplar", "voice_exemplar"):
        by = {}
        for r in store.q("SELECT id, person_id, emb, source FROM %s WHERE deleted=0 AND person_id IS NOT NULL AND "
                         "source != 'do_not_learn'" % table):
            if r["emb"]:
                by.setdefault(r["person_id"], []).append(r)
        for rows in by.values():
            vs = [from_blob(r["emb"]).astype(np.float32) for r in rows]
            trusted = [i for i, r in enumerate(rows) if r["source"] != "auto"]
            if len(trusted) < min_trusted or len(trusted) == len(rows) or len({v.shape[0] for v in vs}) > 1:
                continue
            E = np.stack(vs)
            c = E[trusted].mean(0)
            s = E @ (c / (np.linalg.norm(c) or 1.0))
            cut = max(floor, float(np.median(s[trusted]) - z * s[trusted].std()))
            for r, si in zip(rows, s):
                if r["source"] == "auto" and si < cut and store.tombstone(table, r["id"]):
                    n += 1
    return n


def forget_episodes(store, now, age_days=60, max_importance=2):
    """Step 7: low-importance, never-retrieved, old episodes are tombstoned (summaries and reflections are kept)."""
    rows = store.q("SELECT id FROM episode WHERE deleted=0 AND importance <= ? AND access_count = 0 AND t_end < ? "
                   "AND kind NOT IN ('summary','reflection')", (max_importance, now - age_days * DAY))
    for r in rows:
        store.tombstone("episode", r["id"])
    return len(rows)


def update_object_beliefs(store, now, keep_days=14):
    """Step 8: fold object sightings into beliefs (last place + place histogram), then prune old sightings."""
    beliefs = {r["label"]: r for r in store.q("SELECT * FROM object_belief WHERE deleted=0")}
    since = max([b["last_seen"] or 0 for b in beliefs.values()] or [0])
    changed = set()
    for s in store.q("SELECT label, place_id, x, y, t FROM object_sighting WHERE deleted=0 AND t > ? ORDER BY t",
                     (since,)):
        b = beliefs.get(s["label"]) or {"label": s["label"], "place_hist": "{}"}
        hist = json.loads(b.get("place_hist") or "{}")
        if s["place_id"]:
            hist[s["place_id"]] = hist.get(s["place_id"], 0) + 1
        b.update(last_place_id=s["place_id"] or b.get("last_place_id"), last_xy=json.dumps([s["x"], s["y"]]),
                 last_seen=s["t"], place_hist=json.dumps(hist))
        beliefs[s["label"]] = b
        changed.add(s["label"])
    for b in (beliefs[k] for k in changed):              # untouched beliefs keep their hlc: no re-sync
        store.put("object_belief", {k: b[k] for k in ("id", "label", "last_place_id", "last_xy", "last_seen",
                                                      "place_hist") if k in b})
    old = store.q("SELECT id FROM object_sighting WHERE deleted=0 AND t < ?", (now - keep_days * DAY,))
    for r in old:
        store.tombstone("object_sighting", r["id"])
    return len(old)


async def rewrite_cards(store, llm, since):
    """Step 4: regenerate the <=600-char core card of every person whose facts changed (any write: added, retracted,
    expired, forgotten) or who got a reflection since `since`, from active facts plus recent reflections. The previous
    card is kept in kv `card_prev.<pid>`; a person with no active facts left loses the card."""
    pids = {r["subject_id"] for r in store.q(
        "SELECT DISTINCT f.subject_id FROM fact f JOIN person p ON p.id = f.subject_id "
        "WHERE f.hlc > ? AND p.deleted = 0 AND p.do_not_learn = 0", (wall_ms_prefix(since),))}
    refl = {}
    for r in store.q("SELECT people, text, t_end FROM episode WHERE deleted=0 AND kind='reflection' AND t_end > ? "
                     "ORDER BY t_end DESC", (since - 30 * DAY,)):
        for pid in json.loads(r["people"] or "[]"):
            refl.setdefault(pid, []).append(r["text"])
            if r["t_end"] > since:
                pids.add(pid)
    n = 0
    for pid in sorted(pids):
        p = store.get("person", pid)
        facts = store.facts_about(pid, limit=40)
        if not p or p.get("do_not_learn"):
            continue
        if not facts:                                   # all retracted/expired: the card must not keep them alive
            if p.get("card"):
                store.put("person", {"id": pid, "card": None})
                n += 1
            continue
        lines = ["- " + f["text"] for f in facts] + ["- (insight) " + t for t in refl.get(pid, [])[:5]]
        card = await llm.complete(P.CARD.format(name=p["display_name"] or pid, facts="\n".join(lines)),
                                  max_tokens=220)
        if card and card != p.get("card"):
            if p.get("card"):
                store.kv_set("card_prev." + pid, {"card": p["card"], "t": time.time()})
            store.put("person", {"id": pid, "card": card[:600]})
            n += 1
    return n


async def run(mirror, extractor, llm, budget_s=600):
    """One consolidation pass; returns a stats dict. Stops early when the time budget runs out."""
    store, now, t0 = mirror.store, time.time(), time.monotonic()
    last = store.kv_get("consolidate.last", 0) or 0
    stats = {"backlog": await extractor.backlog()}
    for name, fn in (("expired", expire_facts), ("forgotten", forget_episodes), ("sightings_pruned",
                                                                                   update_object_beliefs)):
        stats[name] = await asyncio.to_thread(fn, store, now)
    stats["deduped"] = await asyncio.to_thread(dedupe_facts, store)
    stats["skills"] = await asyncio.to_thread(skills.mine, store, now)
    stats["decayed"] = await asyncio.to_thread(decay_confidence, store, now)
    stats["routines"] = await asyncio.to_thread(routines.update, store, now)
    stats["raw_dropped"] = await asyncio.to_thread(drop_raw, store, now)
    stats["exemplars_pruned"] = await asyncio.to_thread(prune_exemplars, store)
    for name, step in (("summaries", lambda: summarise(store, llm, mirror.embed, now)),
                       ("reflections", lambda: reflect(mirror, llm, now)),
                       ("cards", lambda: rewrite_cards(store, llm, last))):
        if time.monotonic() - t0 < budget_s:
            stats[name] = await step()
    store.kv_set("consolidate.last", now)
    mirror.retriever.refresh()
    mirror.nudge()
    log.info("consolidation: %s (%.0f s)", stats, time.monotonic() - t0)
    return stats
