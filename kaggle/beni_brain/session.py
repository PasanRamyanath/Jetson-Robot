"""One robot connection (/ws): frame routing and the per-turn pipeline (§10.7 without LangGraph).

Critical path per turn: STT final -> route (regex) -> recall (hybrid retrieval, ~10 ms) -> LLM stream with tools
(max 3 rounds) -> sentence chunks -> TTS -> tts.chunk frames. Extraction runs afterwards in the background.
`interrupt{played_ms}` cancels the turn task (closing the vLLM stream aborts generation) and truncates the history
to the words the user actually heard.
Speculative start (§8.6 #1): a fully stable STT partial starts recall + LLM round 0 early; its output is held back
and used only if the final transcript has the same words.
"""
import asyncio
import base64
import collections
import json
import logging
import re
import time
import uuid

from websockets.exceptions import ConnectionClosed

from beni_common.memory import skills
from beni_common.memory.retrieval import format_context
from beni_common.schemas import pack, unpack

from . import prompts as P
from .stt import OpusDecoder, pcm16_to_f32
from .tools import Tools
from .tts import EmoTag, Splitter, duration_ms

log = logging.getLogger("session")
REFLEX = re.compile(r"^\W*(stop|halt|freeze|wait)( it| now| beni| please)?\W*$", re.I)
EXPRESSION = {"happy": "happy", "excited": "happy", "sad": "sad", "apologetic": "sad", "curious": "curious"}
VISION = re.compile(r"\b(look|see|seeing|wearing|holding|colou?r|what(?:'s| is) (?:this|that|it)|show)\b", re.I)


def truncate(spoken, played_ms):
    """[(text, ms)] + ms actually played -> the text the user heard."""
    out, acc = [], 0
    for text, ms in spoken:
        if acc >= played_ms:
            break
        if acc + ms <= played_ms:
            out.append(text)
        else:
            words = text.split()
            out.append(" ".join(words[:int(len(words) * (played_ms - acc) / max(ms, 1))]) + "...")
        acc += ms
    return " ".join(out)


def _norm(text):
    return " ".join(re.sub(r"[^\w\s']", " ", text.lower()).split())


class Spec:
    """§8.6 #1: round 0 of the LLM started on a fully stable STT partial. Output (text and tool calls) is only
    buffered; run_turn replays it if the final transcript matches, else cancels it. Tools never run speculatively."""

    def __init__(self, text):
        self.text, self.key, self.messages = text, _norm(text), None
        self.ready, self.q, self.task = asyncio.Event(), asyncio.Queue(), None

    async def items(self):
        while True:
            item = await self.q.get()
            if item is None:
                return
            if item[0] == "error":
                raise item[1]
            yield item


def _args(call):
    try:
        a = json.loads(call.get("arguments") or "{}")
    except ValueError:
        return {}
    return a if isinstance(a, dict) else {}


def _tag(turn):
    return "<emo=%s> " % turn.emo.emo if turn.emo is not None and turn.emo.emo else ""


def _ok(res):
    return not res.startswith("error") and '"ok": false' not in res and res != "camera unavailable"


class Turn:
    def __init__(self, tid, ctx, stt):
        self.id, self.ctx = tid, ctx or {}
        self.proactive = self.ctx.get("proactive")
        self.stt = stt
        self.audio_done = asyncio.Event()
        self.task = self.partial_task = self.snap_task = None
        self.spoken, self.played_ms, self.emo = [], None, None
        people = [p.get("id") for p in self.ctx.get("people_present") or () if p.get("id")]
        self.speaker = people[0] if len(people) == 1 else None
        self.heard = self.reply = ""
        self.calls = []                                   # [{tool, args, ok}] for the skill miner (§11.9 step 11)
        self.spec = None
        self.t0 = time.monotonic()


class Session:
    def __init__(self, brain, ws, hello):
        self.brain, self.cfg, self.ws, self.hello = brain, brain.cfg, ws, hello
        self.seq, self.open = 0, True
        self._send_lock = asyncio.Lock()
        self.turns, self.pending = {}, {}
        self.history = collections.deque(maxlen=2 * self.cfg.history_turns)
        self.tools = Tools(self)
        self.opus = None
        self.sync_task = None
        self.last_fb = None                          # feedback row of the previous turn (§11.12)
        self.done = None                             # (turn, history entry) of the last reply still playing out

    # ------------------------------------------------------------------ transport
    async def send(self, type_, **kw):
        if not self.open:
            return False
        kw.setdefault("turn", 0)
        async with self._send_lock:
            self.seq += 1
            kw.update(type=type_, seq=self.seq, t=time.time())
            try:
                await self.ws.send(pack(kw))
                return True
            except ConnectionClosed:
                self.open = False
                return False

    async def run(self):
        sync = self.brain.mirror.attach(self.send, lambda: self.open)
        self.sync_task = asyncio.create_task(sync.run())
        try:
            async for raw in self.ws:
                if isinstance(raw, str):
                    continue
                try:
                    await self.on_frame(unpack(raw))
                except Exception:
                    log.exception("frame")
        except ConnectionClosed:
            pass
        finally:
            self.open = False
            self.sync_task.cancel()
            for t in list(self.turns.values()):
                if t.task:
                    t.task.cancel()
            for f in self.pending.values():
                f.cancel()

    async def on_frame(self, f):
        t, tid = f.get("type"), f.get("turn", 0)
        if t == "audio.chunk":
            turn = self.turns.get(tid)
            if turn is not None:
                turn.stt.add(self._decode(f))
                self._maybe_partial(turn)
        elif t == "turn.begin":
            self.begin(tid, f.get("context"))
        elif t == "audio.end":
            turn = self.turns.get(tid)
            if turn is not None:
                if not f.get("proactive"):
                    turn.proactive = None
                turn.audio_done.set()
        elif t == "interrupt":
            self.interrupt(tid, f.get("played_ms") or 0)
        elif t == "action.result":
            self._resolve(f.get("call_id"), f)
        elif t == "vision.snapshot":
            self._resolve(("snap", f.get("req")), f)
        elif t in ("memory.delta", "memory.ack"):
            await self.brain.mirror.on_frame(f)
        elif t == "heartbeat":
            if f.get("rtt_probe") is not None:
                await self.send("heartbeat", echo=f["rtt_probe"])
        elif t == "replay.flag":
            asyncio.create_task(self.replay_flag(f))
        elif t == "brain.stop":
            self.brain.request_stop(f.get("reason") or "robot")

    def _decode(self, f):
        if f.get("opus") is not None:
            if self.opus is None:
                self.opus = OpusDecoder()
            return self.opus.decode(f["opus"])
        return pcm16_to_f32(f.get("pcm16") or b"")

    def _resolve(self, key, f):
        fut = self.pending.pop(key, None)
        if fut is not None and not fut.done():
            fut.set_result(f)

    # ------------------------------------------------------------------ robot calls
    async def action(self, name, args, turn, timeout=None):
        cid = uuid.uuid4().hex[:12]
        fut = self.pending[cid] = asyncio.get_running_loop().create_future()
        try:
            if not await self.send("action", turn=turn, name=name, args=args or {}, call_id=cid):
                return {"ok": False, "error": "robot offline"}
            f = await asyncio.wait_for(fut, timeout or self.cfg.action_timeout_s)
            res = f.get("result")
            return res if isinstance(res, dict) else {"ok": bool(f.get("ok")), "result": res}
        except asyncio.TimeoutError:
            return {"ok": False, "error": "timeout"}
        finally:
            self.pending.pop(cid, None)

    async def snapshot(self, cam, turn, timeout=4.0):
        req = uuid.uuid4().hex[:12]
        fut = self.pending[("snap", req)] = asyncio.get_running_loop().create_future()
        try:
            if not await self.send("vision.request", turn=turn, cam=cam, req=req):
                return None
            f = await asyncio.wait_for(fut, timeout)
            return f if f.get("jpeg") else None
        except asyncio.TimeoutError:
            return None
        finally:
            self.pending.pop(("snap", req), None)

    async def replay_flag(self, f, max_want=5):
        """§11.9 step 9: a sleep-replay keyframe -> Florence-2 dense caption + a look for objects Beni was asked
        about and couldn't find -> `replay.result` (the robot adds it to the episode)."""
        v, jpeg = self.brain.vision, f.get("jpeg")
        cap, found = "", []
        if jpeg:
            try:
                cap = await asyncio.to_thread(v.caption, jpeg)
                for w in (f.get("want") or [])[:max_want]:
                    hits, _ = await asyncio.to_thread(v.find, jpeg, w)
                    if hits:
                        found.append(w)
            except Exception as e:
                log.warning("replay flag %s: %s", f.get("id"), e)
        await self.send("replay.result", id=f.get("id"), caption=cap, found=found)

    # ------------------------------------------------------------------ turns
    def begin(self, tid, ctx):
        old = self.turns.get(tid)
        if old is not None and old.task:
            old.task.cancel()
        turn = self.turns[tid] = Turn(tid, ctx, self.brain.stt.new_session(prompt=self._stt_prompt(ctx)))
        if (ctx or {}).get("persons_in_view") and not turn.proactive:
            turn.snap_task = asyncio.create_task(self.snapshot(0, tid))   # in parallel with the user speaking
        turn.task = asyncio.create_task(self.run_turn(turn))
        self.brain.touch()

    def _stt_prompt(self, ctx):
        """Personalised Whisper prompt: names of the people present and home places (§11.8.6)."""
        names = [p.get("name") for p in (ctx or {}).get("people_present") or () if p.get("name")]
        return ("Beni, " + ", ".join(names) + ".") if names else "Beni."

    def _maybe_partial(self, turn):
        if turn.partial_task is None or turn.partial_task.done():
            if turn.stt.pending_s() >= 0.5 and not turn.audio_done.is_set():
                turn.partial_task = asyncio.create_task(self._partial(turn))

    async def _partial(self, turn):
        r = await asyncio.to_thread(turn.stt.step)
        if r and r.get("partial"):
            await self.send("stt.partial", turn=turn.id, text=r["partial"], stable=r.get("stable", ""),
                            lang=r.get("lang"))
            if r.get("stable") == r["partial"]:          # two decodes agree on every word: the user has paused
                self._speculate(turn, r["partial"])

    def _speculate(self, turn, text):
        if (not self.cfg.speculate or turn.proactive or turn.audio_done.is_set() or len(text.split()) < 2
                or REFLEX.match(text) or VISION.search(text)):
            return
        if turn.spec is not None:
            if turn.spec.key == _norm(text):
                return
            turn.spec.task.cancel()
        spec = turn.spec = Spec(text)
        spec.task = asyncio.create_task(self._spec_run(turn, spec))

    async def _spec_run(self, turn, spec):
        try:
            memory = await asyncio.to_thread(self.recall, spec.text, turn)
            spec.messages = self._messages(turn, spec.text, memory)
            spec.ready.set()
            async for item in self.brain.llm.stream(list(spec.messages), P.TOOLS if self.cfg.max_tool_rounds else None):
                spec.q.put_nowait(item)
        except asyncio.CancelledError:
            raise
        except Exception as e:
            spec.q.put_nowait(("error", e))
        finally:
            spec.ready.set()
            spec.q.put_nowait(None)

    async def _take_spec(self, turn, text):
        """The speculative round 0 if it was started on this very transcript, else None (and it is cancelled)."""
        spec, turn.spec = turn.spec, None
        if spec is None:
            return None
        if spec.key != _norm(text):
            spec.task.cancel()
            return None
        await spec.ready.wait()
        if spec.messages is None:
            spec.task.cancel()
            return None
        log.info("turn %s: speculative start used", turn.id)
        return spec

    def _messages(self, turn, user_text, memory, image=None):
        content = user_text if image is None else [
            {"type": "text", "text": user_text},
            {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64," + base64.b64encode(image).decode()}}]
        return ([{"role": "system", "content": P.system(self.cfg, turn.ctx, memory)}] + list(self.history)
                + [{"role": "user", "content": content}])

    def interrupt(self, tid, played_ms):
        if tid not in self.turns and self.done is not None and self.done[0].id == tid:
            self._truncate_done(played_ms)             # the usual case: generated in full, cut off while playing
            return
        for turn in ([self.turns[tid]] if tid in self.turns else [] if tid else list(self.turns.values())):
            turn.played_ms = played_ms
            if turn.task and not turn.task.done():
                turn.task.cancel()

    def _truncate_done(self, played_ms):
        """A barge-in on a finished turn: history keeps only the words the user heard; the feedback row gets -0.5."""
        (turn, entry), self.done = self.done, None
        if played_ms < sum(ms for _, ms in turn.spoken):
            turn.reply = entry["content"] = truncate(turn.spoken, played_ms) or "..."
        if turn.heard and not turn.proactive and self.last_fb:
            self.brain.mirror.store.put("feedback", {"id": self.last_fb, "response": _tag(turn) + turn.reply,
                                                     "signal": -0.5})

    async def run_turn(self, turn):
        try:
            await turn.audio_done.wait()
            if turn.proactive:
                user_text = P.proactive_instruction(turn.proactive, turn.ctx)
            else:
                if turn.partial_task is not None:
                    await asyncio.gather(turn.partial_task, return_exceptions=True)
                r = await asyncio.to_thread(turn.stt.step, True) or {}
                turn.heard = user_text = (r.get("final") or "").strip()
                await self.send("stt.final", turn=turn.id, text=user_text, lang=r.get("lang"), speaker=turn.speaker)
                if not user_text:
                    await self.send("turn.end", turn=turn.id, reason="empty")
                    return
                if REFLEX.match(user_text):
                    asyncio.create_task(self.action("stop", {}, turn.id))
                    await self.speak(turn, ["Okay, stopping."], "calm")
                    turn.reply = "Okay, stopping."
                    await self.send("turn.end", turn=turn.id)
                    return
            spec = None if turn.proactive else await self._take_spec(turn, user_text)
            if spec is not None:
                messages = spec.messages
            else:
                image = await self._vision_image(turn, user_text)
                memory = await asyncio.to_thread(self.recall, user_text, turn)
                messages = self._messages(turn, user_text, memory, image)
            turn.emo = EmoTag()
            turn.reply = await self.respond(turn, messages, spec)
            await self.send("turn.end", turn=turn.id)
        except asyncio.CancelledError:
            if turn.played_ms is not None and turn.spoken:
                turn.reply = truncate(turn.spoken, turn.played_ms)
        except Exception as e:
            log.exception("turn %s", turn.id)
            await self.send("error", turn=turn.id, error=str(e)[:200])
            await self.send("turn.end", turn=turn.id, reason="error")
        finally:
            self.turns.pop(turn.id, None)
            if turn.spec is not None:
                turn.spec.task.cancel()
            if turn.snap_task:
                turn.snap_task.cancel()
            self._remember(turn)
            log.info("turn %s done in %.2f s: %r -> %r", turn.id, time.monotonic() - turn.t0, turn.heard, turn.reply)

    async def _vision_image(self, turn, text):
        if turn.proactive or not VISION.search(text):
            return None
        snap = None
        if turn.snap_task is not None:
            try:
                snap = await asyncio.wait_for(asyncio.shield(turn.snap_task), 1.5)
            except asyncio.TimeoutError:
                pass
        if snap is None:
            snap = await self.snapshot(0, turn.id, timeout=2.5)
        return snap["jpeg"] if snap else None

    def recall(self, query, turn):
        ids = [p["id"] for p in turn.ctx.get("people_present") or () if p.get("id")]
        r = self.brain.mirror.retriever
        r.refresh()
        text = format_context(r.context(query, people=ids, speaker=turn.speaker))
        learned = skills.relevant(self.brain.mirror.store, query)
        if learned:
            text += "\n\n## What worked before\n" + "\n".join("- " + x for x in learned)
        return text

    def _remember(self, turn):
        if turn.heard:
            self.history.append({"role": "user", "content": turn.heard})
        self.done = None
        if turn.reply:
            self.history.append({"role": "assistant", "content": turn.reply})
            if turn.played_ms is None and turn.spoken:
                self.done = (turn, self.history[-1])
        if turn.heard and turn.reply and not turn.proactive:
            prev, self.last_fb = self.last_fb, self.brain.mirror.store.put("feedback", {
                "t": time.time(), "person_id": turn.speaker, "kind": "turn", "prompt": turn.heard,
                "response": _tag(turn) + turn.reply,
                "signal": -0.5 if turn.played_ms is not None else 0.0})               # -0.5: barge-in
            self.brain.spawn(self.brain.extractor.after_turn(turn.heard, turn.reply, turn.ctx, turn.speaker,
                                                             prev=prev))
            if turn.calls:
                skills.log_action(self.brain.mirror.store, turn.heard, turn.calls, turn.speaker)
        self.brain.touch()

    async def respond(self, turn, messages, spec=None):
        emo, sp, q = turn.emo, Splitter(), asyncio.Queue()
        tts = asyncio.create_task(self._tts_worker(turn, q, emo))
        reply = []

        async def out(s):
            if s:
                reply.append(s)
                await self.send("llm.delta", turn=turn.id, text=s)
                for c in sp.feed(s):
                    q.put_nowait(c)
        try:
            for rnd in range(self.cfg.max_tool_rounds + 1):
                calls, said = None, len(reply)
                stream = spec.items() if rnd == 0 and spec is not None else self.brain.llm.stream(
                    messages, P.TOOLS if rnd < self.cfg.max_tool_rounds else None)
                async for kind, val in stream:
                    if kind == "text":
                        await out(emo.feed(val))
                    else:
                        calls = val
                if not calls:
                    break
                messages.append({"role": "assistant", "content": "".join(reply[said:]) or None, "tool_calls": [
                    {"id": c["id"], "type": "function",
                     "function": {"name": c["name"], "arguments": c["arguments"] or "{}"}} for c in calls]})
                results = await asyncio.gather(*(self.tools.execute(c, turn.id, turn.speaker) for c in calls))
                images = []
                for c, (res, imgs) in zip(calls, results):
                    messages.append({"role": "tool", "tool_call_id": c["id"], "content": res})
                    images += imgs
                    turn.calls.append({"tool": c["name"], "args": _args(c), "ok": _ok(res)})
                if images:
                    messages.append({"role": "user", "content": [{"type": "text", "text": "(camera image)"}] + [
                        {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64," +
                                                           base64.b64encode(j).decode()}} for j in images[:2]]})
            await out(emo.flush())
            for c in sp.flush():
                q.put_nowait(c)
            q.put_nowait(None)
            await tts
        finally:
            if not tts.done():
                tts.cancel()                             # barge-in or an LLM error: no worker left waiting on q
            if spec is not None:
                spec.task.cancel()                       # a barge-in or error must not leave it generating
        return "".join(reply).strip()

    async def speak(self, turn, chunks, emo=None):
        q = asyncio.Queue()
        for c in chunks:
            q.put_nowait(c)
        q.put_nowait(None)
        tag = EmoTag()
        tag.emo = emo
        await self._tts_worker(turn, q, tag)

    async def _tts_worker(self, turn, q, emo):
        first = True
        while True:
            text = await q.get()
            if text is None:
                break
            pcm = await asyncio.to_thread(self.brain.tts.synth, text, emo.emo)
            if not pcm:
                continue
            turn.spoken.append((text, duration_ms(pcm)))
            kw = {"emo": EXPRESSION[emo.emo]} if first and emo.emo in EXPRESSION else {}   # eye expression
            await self.send("tts.chunk", turn=turn.id, pcm16_24k=pcm, final=False, **kw)
            first = False
        await self.send("tts.chunk", turn=turn.id, pcm16_24k=b"", final=True)
