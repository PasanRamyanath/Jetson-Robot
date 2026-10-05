"""Brain text/audio helpers: emo tags, TTS chunking, interrupt truncation, LocalAgreement-2 partials."""
import threading
from types import SimpleNamespace

import numpy as np

from beni_brain import prompts as P
from beni_brain.session import REFLEX, VISION, truncate
from beni_brain.stt import SR, _Sess, pcm16_to_f32
from beni_brain.tts import EmoTag, Splitter, StubTTS, duration_ms, split_stream, to_pcm16


def feed_all(tag, parts):
    return "".join(tag.feed(p) for p in parts) + tag.flush()


def test_emo_tag_split_across_deltas():
    t = EmoTag()
    assert feed_all(t, ["<e", "mo=hap", "py> Hi", " there"]) == "Hi there" and t.emo == "happy"


def test_emo_tag_absent_or_unknown():
    t = EmoTag()
    assert feed_all(t, ["Hello", " you"]) == "Hello you" and t.emo is None
    t = EmoTag()
    assert feed_all(t, ["<emo=furious>Grr"]) == "Grr" and t.emo is None


def test_splitter_first_chunk_is_a_clause():
    deltas = list("Sure, I can do that for you, just a second. Here is the **plan**: first we go left. ")
    chunks = list(split_stream(deltas))
    assert chunks[0] == "Sure, I can do that for you,"
    assert chunks[1] == "just a second."
    assert chunks[2] == "Here is the plan: first we go left."


def test_splitter_caps_long_text():
    sp = Splitter(max_chars=40)
    out = sp.feed("word " * 30) + sp.flush()
    assert all(len(c) <= 40 for c in out) and " ".join(out).split() == ["word"] * 30


def test_truncate_to_heard_words():
    spoken = [("Hello there.", 1000), ("One two three four.", 2000)]
    assert truncate(spoken, 0) == ""
    assert truncate(spoken, 1000) == "Hello there."
    assert truncate(spoken, 2000) == "Hello there. One two..."
    assert truncate(spoken, 9999) == "Hello there. One two three four."


def test_routing_regexes():
    assert REFLEX.match("Stop!") and REFLEX.match("wait beni") and not REFLEX.match("stop playing music later")
    assert VISION.search("what am I holding?") and not VISION.search("tell me a joke")


def test_audio_helpers():
    pcm = to_pcm16(np.array([0.0, 1.0, -1.0, 2.0], np.float32))
    assert np.frombuffer(pcm, "<i2").tolist() == [0, 32767, -32767, 32767]
    assert np.allclose(pcm16_to_f32(pcm)[:2], [0.0, 32767 / 32768])
    assert duration_ms(StubTTS().synth("Hello there.", "happy")) > 0


def test_prompts_render():
    s = P.system(SimpleNamespace(home_city="Colombo", persona=""), {"people_present": [{"name": "Nimal"}]}, "- fact")
    assert "Colombo" in s and "- fact" in s
    names = {t["function"]["name"] for t in P.TOOLS}
    assert P.PHYSICAL <= names


class FakeWhisper:
    """Hypotheses grow word by word; the last word flickers until the next step."""

    def __init__(self, script):
        self.script, self.i, self.lock = script, 0, threading.Lock()

    def transcribe(self, audio, **kw):
        text = self.script[min(self.i, len(self.script) - 1)]
        self.i += 1
        return [SimpleNamespace(text=text)], SimpleNamespace(language="en", language_probability=0.9)


def test_local_agreement_stable_prefix():
    owner = SimpleNamespace(m=None, lock=threading.Lock())
    owner.m = FakeWhisper(["where are", "where are my kiss", "where are my keys", "where are my keys"])
    s = _Sess(owner, None, "Beni.", None)
    s.add(np.zeros(SR // 4, np.float32))
    assert s.step() is None                                  # < 0.5 s buffered
    s.add(np.zeros(SR, np.float32))
    assert s.step()["stable"] == ""
    assert s.step()["stable"] == "where are"
    r = s.step()
    assert r["stable"] == "where are my" and r["partial"] == "where are my keys"
    assert s.step(final=True)["final"] == "where are my keys"
