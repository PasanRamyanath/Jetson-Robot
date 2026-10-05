"""Proactive behaviour loop (§12.2), runs offline too.

Every 5 s: build candidates from the world state -> filter by rules (quiet hours, 30-min 'not now' cool-down,
battery) -> Thompson-sample one (bandits.py) -> execute (brain phrases it when online, local template + Piper
otherwise) -> reward after 10 s: +1 engaged, 0 ignored, -1 'not now'.
"""
import asyncio
import datetime as dt
import logging
import random
import re
import time

from beni_common.memory import routines

from ..config import in_windows
from ..util import spawn

log = logging.getLogger("proactive")
TICK_S = 5.0
COOLDOWN_S = 30 * 60
GREET_GAP_S = 4 * 3600
REWARD_WINDOW_S = 10.0
MIN_GAP_S = 120            # never two proactive actions within 2 minutes
NOT_NOW = re.compile(r"\b(not now|stop|be quiet|shut up|go away|leave me|later)\b", re.I)

TEMPLATES = {
    "greet": "Hi {name}! Welcome back.",
    "remind": "{name}, a reminder: {what}.",
    "ask_name": "Hello! I don't think we've met. What's your name?",
    "dock": "My battery is getting low, I'm going to charge.",
    "ask_visitor": "{name}, who was the person who visited {when}? I don't think I know them.",
    "routine_nudge": "{name}, we usually have a chat around now. How's your day going?",
    "share_memory": "{name}, a year ago today: {what}",
    "lost_item": "{name}, I think I spotted the {what} {when}.",
}
ASK_VISITOR_MAX_AGE_S = 3 * 86400
ROUTINE_MIN, ROUTINE_RATIO = 2.0, 4.0      # routine_nudge: >= 2 decayed hits in this hour-of-week, 4x the mean
NUDGE_QUIET_S = 3600                       # ... and no interaction with Beni in the last hour
WANDER_IDLE_S, WANDER_GAP_S, WANDER_BATTERY = 20 * 60, 2 * 3600, 60
SILENT = ("idle_wander",)                  # behaviours with nothing to say (and no engagement to reward)


def parse_when(s, now=None):
    """ISO 8601 ('2026-09-27T18:30', '18:30') or relative ('in 20 minutes') -> epoch seconds, or None."""
    now = now or time.time()
    s = (s or "").strip().lower()
    m = re.match(r"in (\d+) ?(min|minute|minutes|h|hour|hours)$", s)
    if m:
        return now + int(m.group(1)) * (3600 if m.group(2).startswith("h") else 60)
    try:
        if re.match(r"^\d{1,2}:\d{2}$", s):
            h, mi = map(int, s.split(":"))
            d = dt.datetime.fromtimestamp(now).replace(hour=h, minute=mi, second=0, microsecond=0)
            t = d.timestamp()
            return t if t > now else t + 86400
        return dt.datetime.fromisoformat(s.replace("z", "")).timestamp()
    except ValueError:
        return None


def when_phrase(t, now):
    """'this morning', 'yesterday afternoon', 'on Tuesday evening'."""
    a, b = time.localtime(t), time.localtime(now)
    days = (dt.date(b.tm_year, b.tm_mon, b.tm_mday) - dt.date(a.tm_year, a.tm_mon, a.tm_mday)).days
    h = a.tm_hour
    part = "morning" if 5 <= h < 12 else "afternoon" if 12 <= h < 17 else "evening" if 17 <= h < 22 else "night"
    if days == 0:
        return "last night" if part == "night" else "this " + part
    if days == 1:
        return "last night" if part == "night" else "yesterday " + part
    return "on %s %s" % (time.strftime("%A", a), part)


class Proactive:
    def __init__(self, cfg, world, voice, bandits, store, act=None, clock=time.time):
        self.cfg, self.world, self.voice, self.b, self.store, self.act = cfg, world, voice, bandits, store, act
        self.clock = clock
        self.cooldown_until = 0.0
        self.last_action_t = 0.0
        self.pending = None         # (arm, person, t_exec)
        self.last_greet = {}
        self.last_nudge = {}        # person -> day ordinal (routine_nudge / share_memory: once a day each)
        self.last_wander = 0.0
        self._year_ago = {}         # (person, day) -> episode row or None

    # ------------------------------------------------------------------ reminders (kv "reminders")
    def add_reminder(self, who, when, what):
        t = parse_when(when, self.clock())
        if t is None:
            return {"ok": False, "error": "could not parse time %r" % when}
        rs = self.store.kv_get("reminders", []) or []
        rs.append({"who": who, "t": t, "what": what, "done": False})
        self.store.kv_set("reminders", rs[-50:])
        return {"ok": True, "at": time.strftime("%Y-%m-%d %H:%M", time.localtime(t))}

    def _due(self, now):
        return [r for r in self.store.kv_get("reminders", []) or [] if not r["done"] and r["t"] <= now]

    def _mark_done(self, rem):
        rs = self.store.kv_get("reminders", []) or []
        for r in rs:
            if r["t"] == rem["t"] and r["what"] == rem["what"]:
                r["done"] = True
        self.store.kv_set("reminders", [r for r in rs if not r["done"] or r["t"] > self.clock() - 86400])

    # ------------------------------------------------------------------ loop
    def candidates(self, now):
        people = self.world.people_present()
        battery = self.world.battery
        if battery is not None and battery < 20:
            return [("dock", None, {})] if not self.world.robot.get("charging") else []
        out = []
        for r in self._due(now):
            who = next((p for p in people if self.world.name(p).lower() == str(r["who"]).lower()), None)
            if who or not people:
                out.append(("remind", who, {"what": r["what"], "rem": r}))
        for p in people:
            if now - self.last_greet.get(p, 0) > GREET_GAP_S:
                out.append(("greet", p, {}))
        ask = self.store.kv_get("ask_unknown")                 # set by sleep replay (perception/replay.py)
        if people and ask and now - ask.get("t", 0) < ASK_VISITOR_MAX_AGE_S:
            out.append(("ask_visitor", people[0], {"when": when_phrase(ask["t"], now), "episode": ask.get("episode")}))
        ctx = self.world.context()
        if ctx["unknown_faces"] and not people:
            out.append(("ask_name", None, {}))
        found = self.store.kv_get("found_objects", []) or []     # sleep replay found something asked for
        if people and found:
            out.append(("lost_item", people[0], {"what": found[0]["label"], "when": when_phrase(found[0]["t"], now),
                                                 "found": found[0]}))
        day = int(now // 86400)
        quiet_since = now - getattr(self.voice, "last_interaction", 0.0)
        for p in people:
            if self.last_nudge.get(p) == day:
                continue
            hits, mean = routines.level(self.store, p, "talk", now)
            if hits >= ROUTINE_MIN and hits >= ROUTINE_RATIO * mean and quiet_since > NUDGE_QUIET_S:
                out.append(("routine_nudge", p, {}))
            ep = self.year_ago(p, now)
            if ep is not None:
                out.append(("share_memory", p, {"what": ep["text"], "episode": ep["id"]}))
        if not people and self._may_wander(now):
            place = self._wander_target()
            if place:
                out.append(("idle_wander", None, {"place": place}))
        return out

    def year_ago(self, person, now):
        """The most important conversation/summary with `person` from this day last year (importance >= 5)."""
        key = (person, int(now // 86400))
        if key not in self._year_ago:
            t = now - 365 * 86400
            rows = self.store.q("SELECT id, text FROM episode WHERE deleted=0 AND kind IN ('conversation','summary') "
                                "AND importance >= 5 AND t_start BETWEEN ? AND ? AND people LIKE ? "
                                "ORDER BY importance DESC LIMIT 1", (t - 43200, t + 43200, '%"' + person + '"%'))
            self._year_ago = {k: v for k, v in self._year_ago.items() if k[1] == key[1]}   # today's, per person
            self._year_ago[key] = rows[0] if rows else None
        return self._year_ago[key]

    def _may_wander(self, now):
        """idle_wander: nobody around for 20 min, not charging-low, cameras on, at most every 2 h."""
        b = self.world.battery
        return (self.act is not None and not getattr(self.world, "privacy", False) and b is not None
                and not getattr(self.world, "estop", False)
                and b >= WANDER_BATTERY and now - getattr(self.world, "last_person_t", now) > WANDER_IDLE_S
                and now - self.last_wander > WANDER_GAP_S and not (self.world.robot or {}).get("moving"))

    def _wander_target(self):
        rows = self.store.q("SELECT name FROM place WHERE deleted=0 AND name IS NOT NULL AND name != ? AND "
                            "(kind IS NULL OR kind != 'dock')", (getattr(self.world, "place", None) or "",))
        return random.choice(rows)["name"] if rows else None

    async def tick(self):
        now = self.clock()
        self._settle(now)
        if self.voice.busy or now < self.cooldown_until or now - self.last_action_t < MIN_GAP_S:
            return None
        lt = time.localtime(now)
        quiet = in_windows(self.cfg.quiet_hours, lt.tm_hour * 60 + lt.tm_min)
        cands = [c for c in self.candidates(now) if not quiet or c[0] in ("remind", "dock")]
        if not cands:
            return None
        forced = [c for c in cands if c[0] in ("remind", "dock")]      # safety/explicit requests skip the bandit
        if forced:
            arm, person, extra = forced[0]
        else:
            by_arm = {c[0]: c for c in cands}
            person = cands[0][1]
            arm, _ = self.b.choose(list(by_arm) + ["rest"], person, now)
            if arm == "rest":
                self.last_action_t = now
                return "rest"
            _, person, extra = by_arm[arm]
        await self.execute(arm, person, extra)
        self.last_action_t = now
        self.pending = None if arm in SILENT else (arm, person, now)
        return arm

    async def execute(self, arm, person, extra):
        name = self.world.name(person) if person else "there"
        if arm == "greet":
            self.last_greet[person] = self.clock()
        if arm == "remind":
            self._mark_done(extra["rem"])
        if arm == "ask_visitor":
            self.store.kv_set("ask_unknown", None)             # ask once
        if arm == "dock" and self.act is not None:
            spawn(self.act("goto_dock", {}))
        if arm in ("routine_nudge", "share_memory"):
            self.last_nudge[person] = int(self.clock() // 86400)
        if arm == "lost_item":
            self.store.kv_set("found_objects", [f for f in self.store.kv_get("found_objects", []) or []
                                                if f != extra["found"]])
        if arm == "idle_wander":                               # look around a known place: refreshes object beliefs
            self.last_wander = self.clock()
            log.info("proactive idle_wander -> %s", extra["place"])
            spawn(self.act("move_to", {"place": extra["place"]}))
            return
        text = TEMPLATES[arm].format(name=name, what=extra.get("what", ""), when=extra.get("when", ""))
        log.info("proactive %s (%s)", arm, person)
        await self.voice.proactive(arm, {"person": person, "text": text, **{k: v for k, v in extra.items()
                                                                             if k not in ("rem", "found")}}, text)

    def _settle(self, now):
        """Engagement reward once the window has passed."""
        if self.pending is None:
            return
        arm, person, t0 = self.pending
        if now - t0 < REWARD_WINDOW_S:
            return
        engaged = self.voice.last_interaction > t0
        self.b.update(arm, 1.0 if engaged else 0.0, person, t0)
        self.pending = None

    def on_user_text(self, text):
        """Called with every final transcript: 'not now' shortly after a proactive action is a -1 and a cool-down."""
        if NOT_NOW.search(text or ""):
            now = self.clock()
            self.cooldown_until = now + COOLDOWN_S
            if self.pending is not None and now - self.pending[2] < 30:
                arm, person, t0 = self.pending
                self.b.update(arm, -1.0, person, t0)
                self.pending = None

    async def run(self):
        while True:
            await asyncio.sleep(TICK_S)
            try:
                await self.tick()
            except Exception:
                log.exception("proactive tick")
