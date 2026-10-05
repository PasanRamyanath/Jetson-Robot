"""§8.6 #1: the LLM starts on a fully stable STT partial and is used only if the final transcript matches."""
import asyncio
import collections
from types import SimpleNamespace

from beni_brain.config import Config
from beni_brain.llm import StubLLM
from beni_brain.session import Session, Turn
from beni_brain.tts import EmoTag, StubTTS


class CountingLLM(StubLLM):
    calls = 0

    async def stream(self, messages, tools=None, **kw):
        CountingLLM.calls += 1
        async for x in super().stream(messages, tools, **kw):
            yield x


def _session():
    s = Session.__new__(Session)
    s.cfg, s.history, s.sent = Config(), collections.deque(), []
    s.brain = SimpleNamespace(llm=CountingLLM(), tts=StubTTS())
    s.recall = lambda text, turn: ""

    async def send(t, **kw):
        s.sent.append(t)
        return True
    s.send = send
    return s


def test_speculative_round0_reused_or_dropped():
    async def go():
        s = _session()
        t = Turn(1, {}, None)
        s._speculate(t, "what time is it")
        await asyncio.sleep(0.05)
        spec = await s._take_spec(t, "What time is it?")          # same words: reused, no second LLM call
        assert spec is not None and CountingLLM.calls == 1 and "llm.delta" not in s.sent   # nothing sent early
        t.emo = EmoTag()
        assert await s.respond(t, spec.messages, spec) == "You said: what time is it."
        assert CountingLLM.calls == 1 and s.sent[-1] == "tts.chunk"

        t2 = Turn(2, {}, None)
        s._speculate(t2, "what time")
        spec2 = t2.spec
        assert await s._take_spec(t2, "what time is it") is None
        await asyncio.sleep(0)
        assert spec2.task.cancelled() or spec2.task.done()

        t3 = Turn(3, {}, None)
        s._speculate(t3, "what am I holding")                   # needs a camera frame: never speculated
        s._speculate(t3, "stop please")
        assert t3.spec is None

    CountingLLM.calls = 0
    asyncio.run(go())


def test_barge_in_after_the_turn_finished_truncates_history():
    """§8.5: the reply is usually fully sent before the user barges in; history and feedback still get cut."""
    rows = {}

    class Store:
        def put(self, table, row):
            key = row.get("id") or "fb%d" % len(rows)
            rows.setdefault(key, {}).update(row, id=key)
            return key

    async def after_turn(*a, **kw):
        pass

    s = _session()
    s.last_fb, s.done, s.turns = None, None, {}
    s.brain = SimpleNamespace(mirror=SimpleNamespace(store=Store()), extractor=SimpleNamespace(after_turn=after_turn),
                              spawn=lambda c: c.close(), touch=lambda: None)
    t = Turn(7, {}, None)
    t.heard, t.reply = "tell me a story", "Once upon a time. There was a robot."
    t.spoken = [("Once upon a time.", 1000), ("There was a robot.", 1000)]
    s._remember(t)
    assert s.history[-1]["content"] == t.reply and rows["fb0"]["signal"] == 0.0
    s.interrupt(8, 500)                                       # some other turn: nothing happens
    assert s.history[-1]["content"] == t.reply
    s.interrupt(7, 1200)
    assert s.history[-1]["content"] == "Once upon a time. ..." and rows["fb0"]["signal"] == -0.5
    assert rows["fb0"]["response"] == "Once upon a time. ..." and s.done is None
