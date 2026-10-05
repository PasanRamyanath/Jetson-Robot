"""Routines (§11.9 step 10, §12.2): per person and activity, 168 hour-of-week bins of decayed counts.

Written by consolidation (brain), read by the proactive loop (Jetson). Activities: 'seen' (sighting episodes) and
'talk' (conversation episodes). Each (person, activity, day, hour) counts once; counts decay x0.97 per week.
"""
import json
import time

import numpy as np

BINS = 168
WEEK = 7 * 86400.0
DECAY = 0.97
ACTIVITY = {"sighting": "seen", "conversation": "talk"}


def hour_of_week(t):
    lt = time.localtime(t)
    return lt.tm_wday * 24 + lt.tm_hour


def rid(person, activity):
    return "rt_%s_%s" % (person, activity)


def update(store, now, since=None):
    """Fold episodes since the last run into the histograms. Returns the number of routine rows written."""
    since = since if since is not None else (store.kv_get("routine.last") or now - 2 * WEEK)
    rows = store.q("SELECT kind, people, t_start FROM episode WHERE deleted=0 AND kind IN (%s) AND t_start > ? "
                   "AND t_start <= ?" % ",".join("?" * len(ACTIVITY)), tuple(ACTIVITY) + (since, now))
    hits = {}
    for r in rows:
        day = int(r["t_start"] // 86400)
        for p in json.loads(r["people"] or "[]"):
            hits.setdefault((p, ACTIVITY[r["kind"]]), set()).add((day, hour_of_week(r["t_start"])))
    for (p, act), cells in hits.items():
        cur = store.get("routine", rid(p, act)) or {}
        h = np.frombuffer(cur["hist"], np.float32).copy() if cur.get("hist") else np.zeros(BINS, np.float32)
        h *= DECAY ** max(0.0, (now - (cur.get("updated") or now)) / WEEK)
        for _, b in cells:
            h[b] += 1.0
        store.put("routine", {"id": rid(p, act), "person_id": p, "activity": act, "hist": h.tobytes(),
                              "n": float(h.sum()), "updated": now})
    store.kv_set("routine.last", now)
    return len(hits)


def level(store, person, activity, t):
    """(count in t's hour-of-week bin, mean over bins), both decayed to t; (0, 0) when there's no routine yet."""
    r = store.get("routine", rid(person, activity))
    if not r or not r.get("hist"):
        return 0.0, 0.0
    h = np.frombuffer(r["hist"], np.float32) * DECAY ** max(0.0, (t - (r.get("updated") or t)) / WEEK)
    return float(h[hour_of_week(t)]), float(h.mean())
