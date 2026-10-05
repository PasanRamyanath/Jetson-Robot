"""Learning from conversations (§11.7), off the critical path: Mem0-style extract -> ADD/UPDATE/DELETE/NOOP.

Provenance rules are applied in code, not by the LLM: inferred facts are capped at 0.6 confidence and sensitive ones
wait for confirmation; told_by_other facts are down-weighted and keep the source person; UPDATE supersedes (history
kept); corrections also become feedback rows for later preference training.
"""
import asyncio
import datetime as dt
import json
import logging
import re
import time

from beni_common.memory import to_blob

from .. import prompts as P

log = logging.getLogger("extract")
_RAW = re.compile(r"USER:\s*(.*?)\nBENI:\s*(.*)", re.S)


class Extractor:
    def __init__(self, llm, mirror):
        self.llm, self.m = llm, mirror
        self.store = mirror.store
        self._lock = asyncio.Lock()                     # one extraction at a time: keeps vLLM free for live turns

    async def _emb(self, text):
        return await asyncio.to_thread(self.m.embed.doc, text)

    def _subject(self, name, speaker):
        n = (name or "").strip().lower()
        if n in ("home", "house", "family"):
            return "home"
        if n in ("beni", "you", "robot"):
            return "beni"
        if n in ("i", "me", "user", "speaker", ""):
            return speaker or "home"
        return self.m.person_id(name) or speaker or "home"

    async def after_turn(self, user, beni, ctx, speaker=None, t=None, prev=None):
        if not (user or "").strip():
            return 0
        async with self._lock:
            try:
                return await self._extract(user, beni, ctx, speaker, t or time.time(), prev=prev)
            except Exception:
                log.exception("extraction failed")
                return 0

    async def _extract(self, user, beni, ctx, speaker, t, raw_id=None, prev=None):
        people = ctx.get("people_present") or []
        names = {p.get("id"): p.get("name") for p in people}
        out = await self.llm.json(P.EXTRACT.format(
            speaker_name=names.get(speaker) or "unknown", speaker_id=speaker or "none",
            people=", ".join(n or "?" for n in names.values()) or "nobody",
            now=dt.datetime.fromtimestamp(t).strftime("%Y-%m-%d %H:%M"), user=user, beni=beni), P.EXTRACT_SCHEMA)
        summary = (out.get("episode_summary") or "").strip() or ("USER: %s / BENI: %s" % (user, beni))[:300]
        importance = float(min(10, max(1, out.get("importance") or 3)))
        ep = self.store.add_episode(summary, kind="conversation", people=[p for p in names if p],
                                    importance=importance, emb=await self._emb(summary), t_start=t, t_end=t,
                                    parent_ids='["%s"]' % raw_id if raw_id else None)
        n = 0
        for f in out.get("facts") or ():
            n += await self._upsert(f, speaker, ep)
        fb = out.get("feedback")
        if isinstance(fb, dict) and fb.get("kind"):
            self._feedback(fb, user, beni, speaker, t, ep, prev)
        self.m.retriever.refresh()
        self.m.nudge()
        return n

    def _feedback(self, fb, user, beni, speaker, t, ep, prev):
        """Praise/complaints/corrections are about Beni's previous reply: re-score that turn row; a correction also
        becomes a DPO row (prompt, rejected reply, the user's correction; the chosen reply is written at training)."""
        sig = {"praise": 1.0}.get(fb["kind"], -1.0)
        old = self.store.get("feedback", prev) if prev else None
        if old is None:
            self.store.put("feedback", {"t": t, "person_id": speaker, "turn_ref": ep, "kind": fb["kind"],
                                        "prompt": user, "response": beni, "better_response": fb.get("about"),
                                        "signal": sig})
            return
        self.store.put("feedback", {"id": prev, "signal": sig})
        if fb["kind"] == "correction":
            self.store.put("feedback", {"t": t, "person_id": speaker, "turn_ref": prev, "kind": "correction",
                                        "prompt": old["prompt"], "response": old["response"],
                                        "better_response": user, "signal": -1.0})

    async def _upsert(self, f, speaker, ep):
        text = (f.get("text") or "").strip()
        if not text or not f.get("predicate"):
            return 0
        subj = self._subject(f.get("subject"), speaker)
        kind = f.get("source_kind") or "said_by_user"
        conf = float(f.get("confidence") or 0.7)
        pred = re.sub(r"\W+", "_", f["predicate"].lower()).strip("_")[:40]
        status = "active"
        if kind == "inferred":
            conf = min(conf, 0.6)
            if any(s in pred for s in P.SENSITIVE):
                status = "pending_confirm"
        elif kind == "told_by_other" and subj != speaker:
            conf *= 0.8
        valid_to = None
        if f.get("valid_to"):
            try:
                valid_to = dt.datetime.fromisoformat(str(f["valid_to"]).replace("Z", "")).timestamp()
            except ValueError:
                pass
        emb = await self._emb(text)
        row = dict(object=str(f.get("object", ""))[:200], text=text[:300], emb=to_blob(emb), confidence=conf,
                   status=status, valid_to=valid_to, source_episode=ep, source_kind=kind, source_person=speaker)
        similar = self.m.similar_facts(subj, emb)
        op, target, d = "ADD", None, {}
        if similar:
            d = await self.llm.json(P.DECIDE.format(
                existing="\n".join("%s: %s" % (s["id"], s["text"]) for s in similar), candidate=text),
                P.DECIDE_SCHEMA, max_tokens=200)
            op, target = (d.get("op") or "NOOP").upper(), d.get("target")
        ids = {s["id"] for s in similar}
        if op == "ADD":
            now = time.time()
            self.store.put("fact", dict(row, subject_id=subj, predicate=pred, valid_from=now, recorded_at=now))
            return 1
        if op == "UPDATE" and target in ids:
            merged = (d.get("merged_text") or text).strip()
            self.store.supersede_fact(target, **dict(row, text=merged, emb=to_blob(await self._emb(merged))))
            return 1
        if op == "DELETE" and target in ids:
            self.store.put("fact", {"id": target, "status": "retracted", "valid_to": time.time()})
            return 1
        if op == "NOOP" and target in ids:              # re-confirmed: the write resets the 90-day decay (§11.9 5c)
            old = self.store.get("fact", target) or {}
            self.store.put("fact", {"id": target, "confidence": max(old.get("confidence") or 0.0, conf)})
        return 0

    async def backlog(self, limit=200):
        """Offline-mode turns logged by the Jetson (kind='conversation_raw') -> extraction (§11.9 step 1)."""
        rows = self.store.q("SELECT id, text, people, t_end FROM episode WHERE kind='conversation_raw' AND deleted=0 "
                            "ORDER BY t_end LIMIT ?", (limit,))
        done = 0
        for r in rows:
            m = _RAW.match(r["text"] or "")
            if m:
                people = [{"id": p} for p in json.loads(r["people"] or "[]")]
                async with self._lock:
                    try:
                        await self._extract(m.group(1), m.group(2), {"people_present": people},
                                            people[0]["id"] if len(people) == 1 else None, r["t_end"], raw_id=r["id"])
                    except Exception:
                        log.exception("backlog %s", r["id"])
                        continue
            self.store.put("episode", {"id": r["id"], "kind": "conversation_extracted"})
            done += 1
        return done
