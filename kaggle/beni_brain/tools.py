"""Tool execution (§10.6). Physical tools become `action` frames the Jetson executes (20 s timeout, awaits
action.result); memory tools run against the mirror and sync back as deltas; take_snapshot returns an image.
"""
import asyncio
import json
import logging
import re
import time

from beni_common.memory import to_blob

from . import vision_tools as V
from .prompts import PHYSICAL

log = logging.getLogger("tools")
CAMS = {"head": 0, "front": 1}


def _ago(t):
    s = time.time() - (t or 0)
    return "%d minutes ago" % (s // 60) if s < 5400 else "%d hours ago" % (s // 3600) if s < 172800 else \
        time.strftime("on %A %d %B", time.localtime(t))


UNITS = {"minute": 60, "hour": 3600, "day": 86400, "week": 604800, "month": 2592000}


def time_window(text, now=None):
    """recall's time_range (§10.6): 'today', 'yesterday', 'last week', 'past 3 days'... -> (t0, t1) or None."""
    now = now or time.time()
    s = (text or "").strip().lower()
    lt = time.localtime(now)
    midnight = time.mktime((lt.tm_year, lt.tm_mon, lt.tm_mday, 0, 0, 0, 0, 0, -1))
    week0 = midnight - lt.tm_wday * 86400
    fixed = {"today": (midnight, now), "yesterday": (midnight - 86400, midnight),
             "this morning": (midnight, midnight + 43200), "last night": (midnight - 21600, midnight + 21600),
             "this week": (week0, now), "last week": (week0 - 604800, week0)}
    if s in fixed:
        return fixed[s]
    m = re.match(r"(?:the )?(?:last|past) (\d+|few|couple of)?\s*(minute|hour|day|week|month)s?$", s)
    if m:
        n = {"few": 3, "couple of": 2, None: 1}.get(m.group(1)) or int(m.group(1))
        return now - n * UNITS[m.group(2)], now
    return None


class Tools:
    def __init__(self, session):
        self.s = session
        self.m = session.brain.mirror

    async def execute(self, call, turn, speaker=None):
        """-> (content_str, [jpeg bytes])."""
        name = call["name"]
        try:
            args = json.loads(call.get("arguments") or "{}")
        except ValueError:
            return "error: arguments were not valid JSON", []
        try:
            if name in PHYSICAL:
                r = await self.s.action(name, args, turn)
                return json.dumps(r, default=str)[:600], []
            fn = getattr(self, "t_" + name, None)
            if fn is None:
                return "error: unknown tool %s" % name, []
            return await fn(args, turn, speaker)
        except Exception as e:
            log.exception("tool %s", name)
            return "error: %s" % e, []

    async def t_take_snapshot(self, args, turn, speaker):
        snap = await self.s.snapshot(CAMS.get(args.get("camera", "head"), 0), turn)
        if not snap:
            return "camera unavailable", []
        dets = ", ".join(sorted({str(d.get("label", d.get("cls"))) for d in snap.get("detections") or ()}))
        return "image attached%s" % (" (detector sees: %s)" % dets if dets else ""), [snap["jpeg"]]

    async def t_locate(self, args, turn, speaker):
        """Florence-2 on a fresh snapshot -> bearing -> look_at (head cam: relative to the neck; front cam: body)."""
        obj, cam = (args.get("object") or "").strip(), CAMS.get(args.get("camera", "head"), 0)
        snap = await self.s.snapshot(cam, turn)
        if not obj or not snap:
            return "camera unavailable" if obj else "what should I look for?", []
        found, (w, h) = await asyncio.to_thread(self.s.brain.vision.find, snap["jpeg"], obj)
        if not found:
            return "I can't see %s from here" % obj, []
        label, box = found[0]
        dpan, dtilt = V.bearing(box, w, h, cam)
        look = {"target": obj, "dpan": round(dpan, 1), "dtilt": round(dtilt, 1)} if cam == 0 else \
            {"target": obj, "pan": round(dpan, 1)}
        r = await self.s.action("look_at", look, turn)
        return "found %s %s (%d in view); look_at: %s" % (label, V.side(dpan), len(found),
                                                          json.dumps(r, default=str)[:200]), []

    async def t_forget_me(self, args, turn, speaker):
        if not speaker:
            return "I'm not sure who is speaking; ask them to look at me first", []
        r = await self.s.action("forget_person", {"person_id": speaker}, turn)
        return json.dumps(r, default=str)[:300], []

    async def t_remember(self, args, turn, speaker):
        fact = (args.get("fact") or "").strip()
        if not fact:
            return "nothing to remember", []
        subj = self.m.person_id(args.get("about")) or speaker or "home"
        now = time.time()
        self.m.store.put("fact", {"subject_id": subj, "predicate": "note", "object": fact[:200], "text": fact[:300],
                                  "emb": to_blob(self.m.embed.doc(fact)), "confidence": 0.95, "status": "active",
                                  "valid_from": now, "recorded_at": now, "source_kind": "said_by_user",
                                  "source_person": speaker})
        self.m.nudge()
        return "saved", []

    async def t_recall(self, args, turn, speaker):
        q = args.get("query") or ""
        about = self.m.person_id(args.get("about"))
        lines = ["fact: " + f["text"] for f in (self.m.store.facts_about(about, 6) if about else [])]
        win = time_window(args.get("time_range"))
        self.m.retriever.refresh()
        eps = self.m.retriever.retrieve(q, speaker=about or speaker, k=24 if win else 6)
        if win:
            eps = [e for e in eps if e["t_start"] <= win[1] and e["t_end"] >= win[0]][:6] or self.m.store.q(
                "SELECT t_end, text FROM episode WHERE deleted = 0 AND t_start <= ? AND t_end >= ? "
                "ORDER BY importance DESC, t_end DESC LIMIT 6", win[::-1])
        for e in eps:
            lines.append("%s: %s" % (_ago(e["t_end"]), e["text"]))
        return "\n".join(lines) or "no memories found", []

    async def t_find_object(self, args, turn, speaker):
        obj = (args.get("object") or "").strip().lower()
        words = [w for w in obj.replace("my ", "").split() if len(w) > 2] or [obj]
        rows = self.m.store.q("SELECT b.label, b.last_seen, b.place_hist, p.name AS place FROM object_belief b "
                              "LEFT JOIN place p ON p.id = b.last_place_id WHERE b.deleted=0 AND (%s) "
                              "ORDER BY b.last_seen DESC LIMIT 3" % " OR ".join(["b.label LIKE ?"] * len(words)),
                              ["%" + w + "%" for w in words])
        out = ["%s last seen %s%s" % (r["label"], _ago(r["last_seen"]), " in the " + r["place"] if r["place"] else "")
               for r in rows]
        if args.get("search_room"):
            r = await self.s.action("search_room", {"object": obj}, turn)
            out.append("room search: %s" % json.dumps(r, default=str)[:200])
        return "; ".join(out) or "I have no memory of seeing that", []
