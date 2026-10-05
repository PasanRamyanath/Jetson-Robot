"""Skill library (§11.9 step 11, Voyager-style): mine repeated successful tool sequences from `action` episodes.

The brain logs one `action` episode per tool-using turn (`log_action`). Its `media_ref` holds
{"key": normalised trigger, "steps": [[tool, args], ...], "ok": bool}. Nightly, `mine` turns the same trigger with the
same successful multi-step sequence, seen at least 3 times, into a `skill` row, and counts later successes and
failures on it. A skill that has failed 3+ times, and at least as often as it worked, is demoted (tombstoned).
`relevant` gives the prompt lines for the skills that match what the user just said.
"""
import json
import re
import time

DAY = 86400.0
MIN_SUPPORT = 3
MIN_STEPS = 2
DEMOTE_FAILS = 3
# Personal one-offs and cosmetics are not part of a reusable procedure.
NOT_STEPS = {"recall", "remember", "forget_me", "enroll_person", "set_reminder", "set_expression", "stop"}
FILLER = {"beni", "please", "can", "could", "would", "will", "you", "hey", "ok", "okay", "the", "a", "an", "me",
          "for", "now", "just", "go", "and", "to", "my", "i", "want", "need", "hi", "hello", "thanks"}
_WORD = re.compile(r"[a-z0-9']+")


def trigger_key(text):
    """'Beni, could you check the kitchen?' -> 'check kitchen'."""
    return " ".join(w for w in _WORD.findall((text or "").lower()) if w not in FILLER)


def steps_of(calls):
    """[{tool, args, ok}] -> the procedure part: [[tool, args], ...] without one-off tools."""
    return [[c["tool"], c.get("args") or {}] for c in calls if c["tool"] not in NOT_STEPS]


def _sig(steps):
    return json.dumps(steps, sort_keys=True, separators=(",", ":"))


def _say(steps):
    return "; ".join("%s(%s)" % (t, ", ".join("%s=%s" % kv for kv in sorted(a.items()))) for t, a in steps)


def log_action(store, trigger, calls, speaker=None, now=None):
    """One tool-using turn -> an `action` episode (low importance: forgotten after 60 days unless retrieved)."""
    key, steps = trigger_key(trigger), steps_of(calls)
    if not key or not steps:
        return None
    ok = all(c.get("ok", True) for c in calls)
    text = "When asked '%s' I did: %s%s" % (trigger.strip()[:120], _say(steps), "" if ok else " (it failed)")
    return store.add_episode(text, kind="action", people=[speaker] if speaker else [], importance=2.0,
                             t_start=now, t_end=now,
                             media_ref=json.dumps({"key": key, "steps": steps, "ok": ok}, sort_keys=True))


def _actions(store, since):
    out = []
    for r in store.q("SELECT id, t_end, media_ref FROM episode WHERE deleted=0 AND kind='action' AND t_end > ? "
                     "ORDER BY t_end", (since,)):
        try:
            a = json.loads(r["media_ref"] or "")
        except ValueError:
            continue
        if isinstance(a, dict) and a.get("key") and isinstance(a.get("steps"), list):
            a["t"] = r["t_end"]
            out.append(a)
    return out


def mine(store, now=None, window_days=30):
    """-> {"new": n, "updated": n, "demoted": n}. Idempotent per night via kv `skills.last`."""
    now = now or time.time()
    last = store.kv_get("skills.last", 0) or 0
    skills = {}
    for s in store.q("SELECT id, trigger, steps, success, fail FROM skill WHERE deleted=0"):
        skills.setdefault(s["trigger"], []).append(s)
    stats = {"new": 0, "updated": 0, "demoted": 0}
    # 1. score the existing skills on what happened since the last pass
    touched = {}
    for a in _actions(store, last):
        for s in skills.get(a["key"], ()):
            if a["ok"] and _sig(a["steps"]) == s["steps"]:
                s["success"] = (s["success"] or 0) + 1
            elif not a["ok"]:
                s["fail"] = (s["fail"] or 0) + 1
            else:
                continue
            touched[s["id"]] = s
    for s in touched.values():
        if s["fail"] >= DEMOTE_FAILS and s["fail"] >= s["success"]:
            store.tombstone("skill", s["id"])
            stats["demoted"] += 1
        else:
            store.put("skill", {"id": s["id"], "success": s["success"], "fail": s["fail"]})
            stats["updated"] += 1
    # 2. propose new skills from the window (never re-propose a demoted one: its tombstone stays in the table)
    known = {(r["trigger"], r["steps"]) for r in store.q("SELECT trigger, steps FROM skill")}
    groups = {}
    for a in _actions(store, now - window_days * DAY):
        if a["ok"] and len(a["steps"]) >= MIN_STEPS:
            groups.setdefault((a["key"], _sig(a["steps"])), []).append(a)
    for (key, sig), runs in groups.items():
        if len(runs) < MIN_SUPPORT or (key, sig) in known:
            continue
        steps = runs[-1]["steps"]
        store.put("skill", {"name": key, "trigger": key, "steps": sig, "success": len(runs), "fail": 0,
                            "description": "When asked to '%s': %s" % (key, _say(steps))})
        stats["new"] += 1
    store.kv_set("skills.last", now)
    return stats


def relevant(store, text, k=2, min_overlap=0.6):
    """Prompt lines for skills whose trigger matches `text` (same key, or mostly the same words)."""
    key = trigger_key(text)
    words = set(key.split())
    if not words:
        return []
    scored = []
    for s in store.q("SELECT trigger, description, success, fail FROM skill WHERE deleted=0"):
        tw = set((s["trigger"] or "").split())
        j = 1.0 if s["trigger"] == key else len(words & tw) / float(len(words | tw) or 1)
        if j >= min_overlap:
            scored.append((j, s["success"] or 0, "%s (worked %d/%d times)" % (
                s["description"], s["success"] or 0, (s["success"] or 0) + (s["fail"] or 0))))
    scored.sort(reverse=True)
    return [line for _, _, line in scored[:k]]
