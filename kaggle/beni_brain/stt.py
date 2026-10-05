"""Streaming STT (§10.5): faster-whisper large-v3-turbo FP16 on GPU1 with LocalAgreement-2 partials, or a stub.

Uplink audio arrives as pcm16 LE 16 kHz (40 ms chunks) or Opus 20 ms frames; both are decoded to float32 here.
"""
import ctypes
import ctypes.util
import threading

import numpy as np

SR = 16000
MAX_WINDOW_S = 25


class StreamingWhisper:
    """LocalAgreement-2 (Machacek et al., whisper_streaming): the stable prefix is the longest common prefix of two
    consecutive hypotheses. step() is blocking: call it through asyncio.to_thread."""

    def __init__(self, model_dir, device="cuda", compute_type="float16"):
        from faster_whisper import WhisperModel
        self.m = WhisperModel(model_dir, device=device, compute_type=compute_type)
        self.lock = threading.Lock()                # one decode at a time on the GPU
        self.m.transcribe(np.zeros(SR, np.float32), beam_size=1, without_timestamps=True)   # warm-up

    def new_session(self, lang=None, prompt=None, hotwords=None):
        return _Sess(self, lang, prompt, hotwords)


class _Sess:
    def __init__(self, owner, lang, prompt, hotwords):
        self.o, self.lang, self.prompt, self.hot = owner, lang, prompt, hotwords
        self.chunks, self.n, self.stepped_n = [], 0, 0
        self.prev = []
        self.buf_lock = threading.Lock()            # add() runs on the loop while step() runs in a worker thread

    def add(self, pcm):
        with self.buf_lock:
            self.chunks.append(pcm)
            self.n += len(pcm)

    def pending_s(self):
        return (self.n - self.stepped_n) / SR

    def _audio(self):
        with self.buf_lock:
            if len(self.chunks) > 1:
                self.chunks = [np.concatenate(self.chunks)]
            a = self.chunks[0] if self.chunks else np.zeros(0, np.float32)
            self.stepped_n = self.n
        return a[-SR * MAX_WINDOW_S:]

    def step(self, final=False):
        if self.n < SR * 0.5 and not final:
            return None
        with self.o.lock:
            segs, info = self.o.m.transcribe(
                self._audio(), language=self.lang, beam_size=3 if final else 1, initial_prompt=self.prompt,
                hotwords=self.hot, vad_filter=False, word_timestamps=False, condition_on_previous_text=False,
                without_timestamps=True, temperature=0.0)
            words = " ".join(s.text for s in segs).split()
        if final:
            return {"final": " ".join(words), "lang": info.language, "conf": info.language_probability}
        n = 0
        while n < min(len(words), len(self.prev)) and words[n] == self.prev[n]:
            n += 1
        self.prev = words
        return {"partial": " ".join(words), "stable": " ".join(words[:n]), "lang": info.language}


class StubSTT:
    """Tests / dry runs: any non-empty utterance transcribes to `self.text`."""

    def __init__(self, text="hello beni"):
        self.text = text

    def new_session(self, lang=None, prompt=None, hotwords=None):
        return _StubSess(self.text)


class _StubSess:
    def __init__(self, text):
        self.text, self.n, self.stepped_n = text, 0, 0

    def add(self, pcm):
        self.n += len(pcm)

    def pending_s(self):
        return 0.0

    def step(self, final=False):
        return {"final": self.text if self.n else "", "lang": "en", "conf": 1.0} if final else None


def load(cfg):
    if not cfg.stt_model:
        return StubSTT()
    return StreamingWhisper(cfg.stt_model, device=cfg.stt_device)


def pcm16_to_f32(b):
    return np.frombuffer(b, "<i2").astype(np.float32) / 32768.0


class OpusDecoder:
    """ctypes libopus decoder (16 kHz mono), mirror of the Jetson's cloud/opus.py encoder."""

    def __init__(self, rate=SR):
        name = ctypes.util.find_library("opus") or "libopus.so.0"
        self.lib = ctypes.CDLL(name)
        self.lib.opus_decoder_create.restype = ctypes.c_void_p
        self.lib.opus_decode.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_int32,
                                         ctypes.POINTER(ctypes.c_int16), ctypes.c_int, ctypes.c_int]
        self.lib.opus_decoder_destroy.argtypes = [ctypes.c_void_p]
        err = ctypes.c_int()
        self.st = self.lib.opus_decoder_create(rate, 1, ctypes.byref(err))
        if err.value != 0:
            raise RuntimeError("opus_decoder_create: %d" % err.value)
        self.max = rate * 120 // 1000
        self.out = (ctypes.c_int16 * self.max)()

    def decode(self, pkt):
        n = self.lib.opus_decode(self.st, pkt, len(pkt), self.out, self.max, 0)
        if n < 0:
            raise ValueError("opus_decode: %d" % n)
        return np.frombuffer(self.out, np.int16, n).astype(np.float32) / 32768.0

    def __del__(self):
        if getattr(self, "st", None):
            self.lib.opus_decoder_destroy(self.st)
