"""Offline replies (§8.6 reflexes, §15.1 offline mode): regex intents first (<5 ms), then llama.cpp Qwen2.5-0.5B.

beni-llm (llama-server) is started by the scheduler only while the brain is unreachable; if it isn't up yet we
answer from a template. Raw turns are logged as `conversation_raw` episodes for extraction at the next session.
"""
import asyncio
import json
import logging
import random
import re
import time
import urllib.request

from ..util import spawn

log = logging.getLogger("offline")

SYSTEM = ("You are Beni, a small friendly home robot in {city}, Sri Lanka. Your cloud brain is offline, so keep "
          "answers to one short spoken sentence. If asked something you can't know, say you'll check when your "
          "big brain is back. Time: {time}. People here: {people}.")

FALLBACK = ["My big brain is napping right now, ask me again in a little while?",
            "I can't think that through right now, but I'll remember you asked."]


def _t(text):
    return lambda m, ctx: text


def _place(m):
    return {"name": re.sub(r"(\s+(please|now|beni|okay|ok))+$", "", m.group("p").strip(" .!?,"), flags=re.I)}


def _where(m, ctx):
    obj = m.group("o").strip(" .!?")
    return ctx["find"](obj) if ctx.get("find") else "I'll check when my big brain is back."


INTENTS = [
    (re.compile(r"^\W*(stop|halt|freeze|wait)\b", re.I), "stop", _t("Okay, stopping.")),
    (re.compile(r"\b(privacy mode off|open your eyes|(turn|switch) on (your )?cameras?|(turn|switch) (your )?cameras? "
                r"(back )?on|cameras? on)\b", re.I), ("camera_privacy", lambda m: {"on": False}),
     _t("Okay, I can see again.")),
    (re.compile(r"\b(privacy mode|close your eyes|(turn|switch) off (your )?cameras?|(turn|switch) (your )?cameras? "
                r"off|cameras? off)\b", re.I), ("camera_privacy", lambda m: {"on": True}),
     _t("Okay, my cameras are off.")),
    (re.compile(r"\bwhat(?:'s| is) the time\b|\bwhat time is it\b", re.I), None,
     lambda m, ctx: "It's %s." % time.strftime("%I:%M %p").lstrip("0")),
    (re.compile(r"\b(what(?:'s| is) the date|what day is (it|today))\b", re.I), None,
     lambda m, ctx: time.strftime("Today is %A, %B %d.")),
    (re.compile(r"\b(go|come) (to )?(your )?(dock|charg\w*|bed)\b", re.I), "goto_dock", _t("Going to my charger.")),
    (re.compile(r"\b(?:this (?:place|room|spot) is|call this(?: place| room| spot)?|"
                r"remember this (?:place |room |spot )?as) (?:the |my |our )?(?P<p>[a-z][\w' ]{1,30})", re.I),
     ("save_place", _place),
     lambda m, ctx: "Okay, this is the %s." % _place(m)["name"]),
    (re.compile(r"\bthis is (?:the |my |our )?(?P<p>(?:[a-z']+ ){0,2}(?:room|kitchen|hall|hallway|garage|garden|"
                r"office|study|balcony|veranda|lounge|toilet|bathroom|bedroom|corridor|porch|lobby|workshop))\b",
                re.I),                                  # bare "this is ...": only with a room word ("this is great")
     ("save_place", _place),
     lambda m, ctx: "Okay, this is the %s." % _place(m)["name"]),
    (re.compile(r"\bgo (?:to|into) (?:the |my |our )?(?P<p>(?!sleep\b)[a-z][\w' ]{1,30})", re.I),
     ("move_to", lambda m: {"place": _place(m)["name"]}), lambda m, ctx: "Heading to the %s." % _place(m)["name"]),
    (re.compile(r"\bcome (here|to me)\b", re.I), "come_here", _t("Coming!")),
    (re.compile(r"\bfollow me\b", re.I), ("follow_person", lambda m: {"name": "me"}), _t("Following you.")),
    (re.compile(r"\b(?:where(?:'s| is| are)|have you seen) (?:my |the |our )?(?P<o>[a-z][\w ]{1,30})", re.I), None,
     _where),
    (re.compile(r"\b(battery|charge) (level|left)\b|\bhow much battery\b", re.I), None,
     lambda m, ctx: "My battery is at %s percent." % (ctx["battery"] if ctx.get("battery") is not None
                                                      else "an unknown")),
    (re.compile(r"\b(hi|hello|hey|good (morning|afternoon|evening)|ayubowan)\b", re.I), None,
     lambda m, ctx: "Hello%s!" % (", " + ctx["names"][0] if ctx.get("names") else "")),
    (re.compile(r"\b(thank(s| you)|isthuti)\b", re.I), None, _t("You're welcome!")),
    (re.compile(r"\b(be quiet|shut up|not now|go away)\b", re.I), "quiet", _t("Okay, I'll be quiet.")),
    (re.compile(r"\bforget (about )?me\b", re.I), "forget_request",
     _t("I can forget everything about you. Please confirm when my big brain is back, or say it again to my owner.")),
]


def match_intent(text, ctx=None):
    """-> (action|None, reply, args) or None."""
    ctx = ctx or {}
    for rx, action, reply in INTENTS:
        m = rx.search(text)
        if m:
            args = {}
            if isinstance(action, tuple):
                action, args = action[0], action[1](m)
            return action, reply(m, ctx), args
    return None


WAKING = "Give me a minute to wake up my big brain."


class OfflineBrain:
    def __init__(self, cfg, world=None, act=None, store=None, embed=None):
        self.cfg, self.world, self.act, self.store, self.embed = cfg, world, act, store, embed
        self.history = []
        self.waking = False             # a brain cold start was just requested: say so once (§10.4)

    def _ctx(self):
        if self.world is None:
            return {}
        return {"names": [self.world.name(p) for p in self.world.people_present()], "battery": self.world.battery,
                "find": self.find_object}

    async def __call__(self, text, meta=None):
        ctx = self._ctx()
        hit = match_intent(text, ctx)
        if hit:
            action, reply, args = hit
            if action and self.act is not None:
                spawn(self.act(action, args))
        else:
            reply = await self.llm(text, ctx) or random.choice(FALLBACK)
        if self.waking:
            self.waking, reply = False, WAKING + " " + reply
        await self._log(text, reply)
        return reply

    def find_object(self, obj):
        """object_belief lookup (the same table the brain's find_object reads)."""
        if self.store is None:
            return "I don't remember."
        words = [w for w in obj.lower().split() if len(w) > 2] or [obj.lower()]
        rows = self.store.q("SELECT b.label, b.last_seen, p.name AS place FROM object_belief b LEFT JOIN place p "
                            "ON p.id=b.last_place_id WHERE b.deleted=0 AND (%s) ORDER BY b.last_seen DESC LIMIT 1"
                            % " OR ".join(["b.label LIKE ?"] * len(words)), ["%" + w.rstrip("s") + "%" for w in words])
        if not rows:
            return "I haven't seen a %s." % obj
        r, ago = rows[0], max(0, time.time() - (rows[0]["last_seen"] or 0))
        when = "a moment ago" if ago < 120 else "%d minutes ago" % (ago // 60) if ago < 5400 else \
            "%d hours ago" % (ago // 3600) if ago < 172800 else "%d days ago" % (ago // 86400)
        return "I saw a %s %s%s." % (r["label"], "in the %s " % r["place"] if r["place"] else "", when)

    async def llm(self, text, ctx):
        msgs = [{"role": "system", "content": SYSTEM.format(city=self.cfg.home_city, time=time.strftime("%H:%M"),
                                                             people=", ".join(ctx.get("names") or []) or "unknown")}]
        msgs += self.history[-4:] + [{"role": "user", "content": text}]
        body = json.dumps({"messages": msgs, "max_tokens": 60, "temperature": 0.6, "cache_prompt": True}).encode()
        try:
            r = await asyncio.get_running_loop().run_in_executor(None, _post, self.cfg.llm_url + "/v1/chat/completions",
                                                                 body, 12)
            out = r["choices"][0]["message"]["content"].strip()
        except Exception as e:
            log.info("local llm unavailable: %s", e)
            return None
        out = re.sub(r"<[^>]+>", "", out).split("\n")[0].strip()
        self.history = self.history[-2:] + [{"role": "user", "content": text}, {"role": "assistant", "content": out}]
        return out

    async def _log(self, text, reply):
        if self.store is None:
            return
        people = self.world.people_present() if self.world is not None else []
        emb = None
        try:                                    # ONNX embedding: off the loop, like ASR/TTS
            if self.embed is not None:
                emb = await asyncio.get_running_loop().run_in_executor(None, self.embed.doc, text)
        except Exception:
            pass
        self.store.add_episode("USER: %s\nBENI: %s" % (text, reply), kind="conversation_raw", people=people,
                               importance=3.0, emb=emb)


def _post(url, body, timeout):
    req = urllib.request.Request(url, body, {"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())
