"""Streaming text -> speech (§10.5): <emo=X> tag stripping, low-latency chunking, Kokoro-82M or CosyVoice2 (GPU1).

Output is what the Jetson player expects in tts.chunk.pcm16_24k: 24 kHz mono s16le.
"""
import logging
import re

import numpy as np

from .prompts import EMOS

RATE = 24000
_TAG = re.compile(r"^\s*<emo=(\w+)>\s*")
_ANY_TAG = re.compile(r"<emo=\w*>?")
_END = re.compile(r"[.!?…]+[\"')\]]*\s")
_CLAUSE = re.compile(r"[,;:—–]\s")
_MARKDOWN = re.compile(r"[*_#`~>|]+")
_SPACE_PUNCT = re.compile(r"\s+([,.;:!?…])")


class EmoTag:
    """Strips the leading <emo=X> control tag from a streamed reply; .emo holds X (or None)."""

    def __init__(self, max_wait=24):
        self.buf, self.done, self.emo, self.max_wait = "", False, None, max_wait

    def feed(self, s):
        if self.done:
            return s
        self.buf += s
        m = _TAG.match(self.buf)
        if m:
            self.emo = m.group(1) if m.group(1) in EMOS else None
            self.done, out = True, self.buf[m.end():]
        else:
            head = self.buf.lstrip()[:5]
            if head and "<emo=".startswith(head) and len(self.buf) <= self.max_wait:
                return ""                           # could still become a tag: hold back
            self.done, out = True, self.buf
        self.buf = ""
        return out

    def flush(self):
        out, self.buf, self.done = self.buf, "", True
        return out


def clean(text):
    """Make a chunk speakable: drop stray tags and markdown."""
    return _SPACE_PUNCT.sub(r"\1", " ".join(_MARKDOWN.sub(" ", _ANY_TAG.sub("", text)).split()))


class Splitter:
    """Cut streamed text into speakable chunks ASAP.

    First chunk: may end at a clause boundary once it has >= min_words words (cuts time-to-first-audio);
    later chunks: whole sentences. Anything over max_chars is cut at the last space.
    """

    def __init__(self, min_words=4, max_chars=220):
        self.buf, self.first, self.min_words, self.max_chars = "", True, min_words, max_chars

    def _cut(self):
        m = _END.search(self.buf)
        if m:
            return m.end()
        if self.first:
            for m in _CLAUSE.finditer(self.buf):
                if len(self.buf[:m.end()].split()) >= self.min_words:
                    return m.end()
        if len(self.buf) > self.max_chars:
            return (self.buf.rfind(" ", 0, self.max_chars) + 1) or self.max_chars
        return None

    def feed(self, s):
        self.buf += s
        out = []
        while True:
            cut = self._cut()
            if cut is None:
                return out
            chunk, self.buf = clean(self.buf[:cut]), self.buf[cut:]
            if chunk:
                out.append(chunk)
                self.first = False

    def flush(self):
        chunk, self.buf = clean(self.buf), ""
        return [chunk] if chunk else []


def split_stream(deltas):
    """Generator form (tests / offline use): iterable of text deltas -> speakable chunks."""
    sp = Splitter()
    for d in deltas:
        yield from sp.feed(d)
    yield from sp.flush()


def to_pcm16(audio):
    return (np.clip(np.asarray(audio, np.float32), -1.0, 1.0) * 32767.0).astype("<i2").tobytes()


# Kokoro has no instruct/emotion input: vary the pace a little so the mood still comes through.
SPEED = {"excited": 1.1, "happy": 1.05, "sad": 0.9, "calm": 0.92, "apologetic": 0.95}


class KokoroTTS:
    """hexgrad/Kokoro-82M via the `kokoro` package (KPipeline). ~50-100 ms per sentence on a T4."""

    def __init__(self, voice="af_heart", device="cuda", lang="a"):
        from kokoro import KPipeline
        self.voice = voice
        self.pipe = KPipeline(lang_code=lang, repo_id="hexgrad/Kokoro-82M", device=device)
        self.synth("Hello.")                        # warm-up: loads the voice pack and compiles kernels

    def synth(self, text, emo=None):
        parts = []
        for r in self.pipe(text, voice=self.voice, speed=SPEED.get(emo, 1.0)):
            a = r.audio if hasattr(r, "audio") else r[2]
            if a is not None:
                parts.append(a.detach().cpu().numpy() if hasattr(a, "detach") else np.asarray(a))
        return to_pcm16(np.concatenate(parts)) if parts else b""


# CosyVoice2 "instruct" mode: natural-language style control (§10.5)
EMO_TO_INSTRUCT = {
    "happy": "Speak cheerfully with a bright, warm tone.",
    "excited": "Speak with excitement and energy.",
    "sad": "Speak softly and gently, a little sad.",
    "calm": "Speak calmly and slowly.",
    "curious": "Speak with curiosity, slightly rising tone.",
    "apologetic": "Speak gently and apologetically.",
}


class CosyVoiceTTS:
    """CosyVoice2-0.5B FP16 in Beni's cloned voice (§10.5): instruct2 for emotions, cross-lingual otherwise.

    Needs the CosyVoice repo on sys.path and a reference wav. Handles both repo generations: newer ones take the
    prompt as a path and want <|endofprompt|> in the instruction; older ones take a 16 kHz tensor and add it themselves.
    """

    def __init__(self, repo, model_dir, prompt_wav):
        import inspect
        import sys
        sys.path[:0] = [repo, repo + "/third_party/Matcha-TTS"]
        from cosyvoice.cli.cosyvoice import CosyVoice2
        self.m = CosyVoice2(model_dir, load_jit=False, load_trt=False, fp16=True)
        self.rate = self.m.sample_rate
        src = inspect.getsource(type(self.m.frontend).frontend_instruct2)
        self.suffix = "" if "endofprompt" in src else "<|endofprompt|>"
        self.prompt = prompt_wav
        try:
            self.synth("Hello.")                    # warm-up, and probes which prompt form this checkout takes
        except Exception:
            from cosyvoice.utils.file_utils import load_wav
            self.prompt = load_wav(prompt_wav, 16000)
            self.synth("Hello.")

    def synth(self, text, emo=None):
        instr = EMO_TO_INSTRUCT.get(emo)
        gen = (self.m.inference_instruct2(text, instr + self.suffix, self.prompt, stream=False) if instr else
               self.m.inference_cross_lingual(text, self.prompt, stream=False))
        parts = [o["tts_speech"].float().cpu().numpy().reshape(-1) for o in gen]
        if not parts:
            return b""
        a = np.concatenate(parts)
        if self.rate != RATE:                       # CosyVoice2 is 24 kHz already; keep the wire format fixed
            a = np.interp(np.arange(0, len(a), self.rate / RATE), np.arange(len(a)), a).astype(np.float32)
        return to_pcm16(a)


class StubTTS:
    """Silence sized like speech (~65 ms per character, capped): exercises the pipeline without a GPU."""

    def synth(self, text, emo=None):
        n = int(RATE * min(8.0, 0.065 * len(text)))
        return bytes(2 * n)


def load(cfg):
    if cfg.tts == "stub":
        return StubTTS()
    if cfg.tts == "cosyvoice":
        try:
            return CosyVoiceTTS(cfg.cosyvoice_dir, cfg.cosyvoice_model, cfg.tts_prompt_wav)
        except Exception:                           # one voice throughout: fall back wholesale, never mix (§10.5)
            logging.getLogger("tts").exception("CosyVoice2 unavailable, using Kokoro")
    return KokoroTTS(cfg.tts_voice)


def duration_ms(pcm16):
    return len(pcm16) * 1000 // (2 * RATE)
