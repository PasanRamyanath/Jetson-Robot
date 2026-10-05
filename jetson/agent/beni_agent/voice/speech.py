"""sherpa-onnx wrappers (CPU, 1 thread each): wake word, VAD, local ASR, Piper TTS, speaker embedding.

Model layout under $BENI_MODELS (download with jetson/setup/07_models.sh):
  kws/{tokens.txt,encoder*.onnx,decoder*.onnx,joiner*.onnx,keywords.txt}   zipformer KWS (open vocabulary)
  vad/silero_vad.onnx
  asr/  moonshine-tiny ({preprocess,encode,uncached_decode,cached_decode}*.onnx)
        or whisper-tiny ({*-encoder,*-decoder}.onnx)
  tts/  piper VITS voice ({*.onnx, tokens.txt, espeak-ng-data/})
  spk/  speaker embedding (3dspeaker / wespeaker *.onnx)
Everything is loaded lazily: a missing model disables that feature instead of crashing the agent.
"""
import glob
import logging
import os

import numpy as np

log = logging.getLogger("speech")
SR = 16000


def _one(d, pat):
    hits = sorted(glob.glob(os.path.join(d, pat)))
    int8 = [h for h in hits if ".int8." in h]
    return (int8 or hits or [None])[0]         # prefer int8 exports: ~2x faster on the A57


def _so():
    import sherpa_onnx
    return sherpa_onnx


class WakeWord:
    def __init__(self, d, threshold=0.25, score=1.5):
        so = _so()
        self.kws = so.KeywordSpotter(
            tokens=os.path.join(d, "tokens.txt"), encoder=_one(d, "encoder*.onnx"), decoder=_one(d, "decoder*.onnx"),
            joiner=_one(d, "joiner*.onnx"), keywords_file=os.path.join(d, "keywords.txt"), num_threads=1,
            provider="cpu", keywords_score=score, keywords_threshold=threshold)
        self.stream = self.kws.create_stream()

    def accept(self, pcm):
        """Feed 16 kHz float32; returns the keyword string when detected."""
        self.stream.accept_waveform(SR, pcm)
        while self.kws.is_ready(self.stream):
            self.kws.decode_stream(self.stream)
        r = self.kws.get_result(self.stream)
        if r:
            self.kws.reset_stream(self.stream)
            return r
        return None


class Vad:
    """Silero VAD. Keeps the finished speech segments so local ASR can transcribe them in offline mode."""

    def __init__(self, model, min_silence=0.45, min_speech=0.25, max_speech=15.0):   # 0.45 = the fast endpoint
        so = _so()
        c = so.VadModelConfig()
        c.silero_vad.model = model
        c.silero_vad.min_silence_duration = min_silence
        c.silero_vad.min_speech_duration = min_speech
        c.silero_vad.max_speech_duration = max_speech
        c.sample_rate = SR
        c.num_threads = 1
        self.vad = so.VoiceActivityDetector(c, buffer_size_in_seconds=30)
        self.window = c.silero_vad.window_size
        self._pend = np.zeros(0, np.float32)

    def accept(self, pcm):
        self._pend = np.concatenate([self._pend, pcm])
        while len(self._pend) >= self.window:      # silero wants exact windows
            self.vad.accept_waveform(self._pend[:self.window])
            self._pend = self._pend[self.window:]
        return self.vad.is_speech_detected()

    def segments(self):
        out = []
        while not self.vad.empty():
            out.append(np.asarray(self.vad.front.samples, np.float32))
            self.vad.pop()
        return out

    def reset(self):
        self.vad.reset()
        self._pend = np.zeros(0, np.float32)


class LocalAsr:
    """Offline (non-streaming) ASR over one VAD segment. ~0.1-0.3 RTF on one A57 core for moonshine-tiny int8."""

    def __init__(self, d):
        so = _so()
        tokens = os.path.join(d, "tokens.txt")
        if _one(d, "*cached_decode*.onnx"):
            self.rec = so.OfflineRecognizer.from_moonshine(
                preprocessor=_one(d, "preprocess*.onnx"), encoder=_one(d, "encode*.onnx"),
                uncached_decoder=_one(d, "uncached_decode*.onnx"), cached_decoder=_one(d, "cached_decode*.onnx"),
                tokens=tokens, num_threads=1)
        else:
            self.rec = so.OfflineRecognizer.from_whisper(
                encoder=_one(d, "*encoder*.onnx"), decoder=_one(d, "*decoder*.onnx"), tokens=_one(d, "*tokens.txt"),
                language="en", task="transcribe", num_threads=1)

    def transcribe(self, pcm):
        s = self.rec.create_stream()
        s.accept_waveform(SR, pcm)
        self.rec.decode_stream(s)
        return s.result.text.strip()


class LocalTts:
    """Piper (VITS) voice for offline replies; output resampled to the 24 kHz player rate."""

    def __init__(self, d, speed=1.0):
        so = _so()
        vits = so.OfflineTtsVitsModelConfig(model=_one(d, "*.onnx"), tokens=os.path.join(d, "tokens.txt"),
                                            data_dir=os.path.join(d, "espeak-ng-data"))
        cfg = so.OfflineTtsConfig(model=so.OfflineTtsModelConfig(vits=vits, num_threads=1, provider="cpu"),
                                  max_num_sentences=1)
        self.tts, self.speed = so.OfflineTts(cfg), speed

    def synth(self, text, out_rate=24000):
        a = self.tts.generate(text, sid=0, speed=self.speed)
        return to_pcm16(resample(np.asarray(a.samples, np.float32), a.sample_rate, out_rate))


class SpeakerEmbed:
    def __init__(self, d):
        so = _so()
        self.ex = so.SpeakerEmbeddingExtractor(
            so.SpeakerEmbeddingExtractorConfig(model=_one(d, "*.onnx"), num_threads=1, provider="cpu"))
        self.dim = self.ex.dim

    def __call__(self, pcm):
        s = self.ex.create_stream()
        s.accept_waveform(SR, pcm)
        s.input_finished()
        v = np.asarray(self.ex.compute(s), np.float32)
        return v / (np.linalg.norm(v) + 1e-9)


def resample(x, src, dst):
    """Linear resampler: speech-quality is fine for 22.05k->24k and far cheaper than a polyphase filter here."""
    if src == dst or len(x) == 0:
        return x
    n = int(round(len(x) * dst / src))
    return np.interp(np.linspace(0, len(x) - 1, n), np.arange(len(x)), x).astype(np.float32)


def to_pcm16(x):
    return (np.clip(x, -1.0, 1.0) * 32767.0).astype("<i2").tobytes()


def load(models, cfg=None):
    """Build what exists. Returns dict name -> object (missing ones absent)."""
    out = {}
    for name, ctor, sub in (("kws", WakeWord, "kws"), ("vad", lambda d: Vad(os.path.join(d, "silero_vad.onnx")), "vad"),
                            ("asr", LocalAsr, "asr"), ("tts", LocalTts, "tts"), ("spk", SpeakerEmbed, "spk")):
        d = os.path.join(models, sub)
        if not os.path.isdir(d):
            log.warning("%s: %s missing, feature disabled", name, d)
            continue
        try:
            out[name] = ctor(d)
        except Exception as e:          # wrong/partial download must not take the agent down
            log.error("%s: load failed: %s", name, e)
    return out
