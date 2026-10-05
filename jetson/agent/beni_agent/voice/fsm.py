"""Voice turn state machine (§8.5), asyncio on one core.

IDLE --wake--> LISTENING --VAD end--> THINKING --first tts--> SPEAKING --done--> LISTENING (6 s follow-up) --> IDLE
                   |                     | no cloud / no first audio in 3 s
                   |                     +--> OFFLINE_REPLY (local ASR -> intents/llama.cpp -> Piper) --> LISTENING
SPEAKING --user speech (after the 400 ms echo window)--> interrupt + LISTENING (barge-in)

All model calls that take >5 ms (ASR, TTS) run in the default executor so audio packets never queue behind them.
"""
import asyncio
import enum
import logging
import re
import time

import numpy as np

from ..util import spawn

log = logging.getLogger("voice")

VAD_SILENCE_S = 0.45                   # speech.Vad min_silence: the shortest endpoint
LONG_TURN = re.compile(r"^\s*(tell me (about|a story)|explain|describe|what do you think|let me tell you|so\b|"
                       r"remember when|i was thinking)", re.I)


class St(enum.Enum):
    IDLE = 0
    LISTENING = 1
    THINKING = 2
    SPEAKING = 3
    OFFLINE_REPLY = 4


class Uplink:
    """Batches 16 kHz pcm16 into >=40 ms WS frames (pcm16) or 20 ms Opus packets."""

    def __init__(self, send, codec="pcm16"):
        self.send, self.buf, self.enc = send, bytearray(), None
        if codec == "opus":
            try:
                from ..cloud.opus import Encoder
                self.enc = Encoder(16000, bitrate=24000)
            except Exception as e:
                log.warning("opus unavailable (%s), falling back to pcm16", e)
        self.codec = "opus" if self.enc else "pcm16"
        self.chunk = 640 if self.enc else 1280          # 20 ms / 40 ms of s16 @16k

    async def push(self, turn, pcm16):
        self.buf += pcm16
        while len(self.buf) >= self.chunk:
            b = bytes(self.buf[:self.chunk])
            del self.buf[:self.chunk]
            if self.enc:
                await self.send("audio.chunk", turn=turn, opus=self.enc.encode(b))
            else:
                await self.send("audio.chunk", turn=turn, pcm16=b)

    async def flush(self, turn):
        if self.buf and not self.enc:
            await self.send("audio.chunk", turn=turn, pcm16=bytes(self.buf))
        self.buf.clear()


class VoiceLoop:
    """cloud: object with .online (asyncio.Event) and async send(type, **kw) -> bool.
    offline: async fn(text, ctx) -> reply text.  context: async fn() -> dict for turn.begin.
    events: Publisher-like with .send(topic, **kw).  on_utterance: optional fn(pcm_f32) for voice ID (background).
    """

    def __init__(self, cfg, speech, player, cloud, offline, context, events=None, turnlog=None, on_utterance=None,
                 on_wake=None):
        self.cfg, self.sp, self.player, self.cloud = cfg, speech, player, cloud
        self.offline, self.context, self.events, self.tl = offline, context, events, turnlog
        self.on_utterance, self.on_wake = on_utterance, on_wake
        self.state = St.IDLE
        self.turn = 0
        self.uplink = Uplink(cloud.send, cfg.uplink)
        self.listen_since = 0.0
        self.speech_since = 0.0
        self.speech_active = False
        self.timeout_s = cfg.no_speech_s
        self.endpoint_s = cfg.endpoint_s
        self.quiet_since = None
        self.last_reply_s = 0.0           # how long Beni's last answer played (short -> quick back-and-forth)
        self.cloud_turn = False           # this turn is being answered by the brain
        self.got_audio = False
        self.utt = []                     # float32 chunks of the current utterance (local ASR / voice ID)
        self.muted = False
        self._timers = []
        self.last_interaction = 0.0

    # ------------------------------------------------------------------ helpers
    def _emit(self, name, **kw):
        if self.events is not None:
            self.events.send(b"event", name=name, turn=self.turn, **kw)

    def _set(self, st):
        if st != self.state:
            log.debug("%s -> %s", self.state.name, st.name)
            self.state = st
            self._emit("voice_state", state=st.name.lower())

    def _later(self, delay, coro_fn, *a):
        async def _t():
            await asyncio.sleep(delay)
            await coro_fn(*a)
        t = spawn(_t())
        self._timers.append(t)
        return t

    def _cancel_timers(self):
        cur = asyncio.current_task()      # a timer that ends in start_listening must not cancel itself
        for t in self._timers:
            if t is not cur:
                t.cancel()
        self._timers = []

    @property
    def busy(self):
        return self.state != St.IDLE

    # ------------------------------------------------------------------ audio in
    async def run(self, mic):
        while True:
            pcm, raw = await mic.q.get()
            try:
                await self.on_audio(pcm, raw)
            except Exception:
                log.exception("on_audio")

    async def on_audio(self, pcm, raw):
        if self.muted:
            return
        now = time.time()
        st = self.state
        if st == St.IDLE:
            kws = self.sp.get("kws")
            if kws is not None and kws.accept(pcm):
                await self.start_listening("wake")
            return
        vad = self.sp.get("vad")
        if vad is None:
            return
        speaking = vad.accept(pcm)
        if st == St.LISTENING:
            if self.cloud_turn:
                await self.uplink.push(self.turn, raw)
            self.utt.append(pcm)
            if speaking and not self.speech_active:
                self.speech_active, self.speech_since = True, now
                self._emit("speech_start")
            if speaking or not self.speech_active:
                self.quiet_since = None
            elif self.quiet_since is None:     # the VAD already waited VAD_SILENCE_S; hold for the rest (§8.6 #2)
                self.quiet_since = now
            if self.speech_active and (now - self.speech_since > self.cfg.max_utterance_s or (
                    self.quiet_since is not None and now - self.quiet_since >= self.endpoint_s - VAD_SILENCE_S)):
                self.speech_active = False
                await self.end_of_speech()
            elif not self.speech_active and now - self.listen_since > self.timeout_s:
                await self.go_idle("no_speech")
        elif st == St.SPEAKING:
            if speaking and not self.speech_active:
                self.speech_active, self.speech_since = True, now
            elif not speaking:
                self.speech_active = False
            if (self.speech_active and now - self.player.started_at > self.cfg.echo_ignore_s
                    and now - self.speech_since >= self.cfg.barge_min_speech_s):
                await self.barge_in()

    # ------------------------------------------------------------------ transitions
    async def start_listening(self, reason, timeout=None):
        self._cancel_timers()
        if reason != "follow_up" or self.state != St.LISTENING:
            self.turn += 1
        self._set(St.LISTENING)
        self.listen_since = time.time()
        self.timeout_s = timeout or (self.cfg.follow_up_s if reason == "follow_up" else self.cfg.no_speech_s)
        self.speech_active, self.utt, self.got_audio = False, [], False
        self.quiet_since = None
        fast = reason == "follow_up" and 0 < self.last_reply_s < 4.0
        self.endpoint_s = self.cfg.endpoint_fast_s if fast else self.cfg.endpoint_s
        vad = self.sp.get("vad")
        if vad is not None:
            vad.reset()
        self.uplink.buf.clear()
        if self.tl:
            self.tl.begin(self.turn, reason=reason)
            self.tl.mark("wake")
        self._emit("listening", reason=reason)
        if reason == "wake" and self.on_wake is not None:
            self.on_wake()
        self.cloud_turn = self.cloud.online.is_set()
        if self.cloud_turn:
            ctx = await self.context()
            self.cloud_turn = await self.cloud.send("turn.begin", turn=self.turn, context=ctx)

    async def end_of_speech(self):
        turn = self.turn
        self._set(St.THINKING)
        self.last_interaction = time.time()
        if self.tl:
            self.tl.mark("speech_end")
        self._emit("speech_end")
        pcm = np.concatenate(self.utt) if self.utt else np.zeros(0, np.float32)
        self.utt = []
        if self.on_utterance is not None and len(pcm) >= 16000 * 1.5:
            self.on_utterance(pcm)
        if self.cloud_turn and self.cloud.online.is_set():
            await self.uplink.flush(turn)
            await self.cloud.send("audio.end", turn=turn)
            self._later(self.cfg.filler_after_s, self._filler, turn)
            self._later(self.cfg.offline_after_s, self._cloud_timeout, turn, pcm)
        else:                             # in the background: the mic keeps flowing (barge-in, no stale backlog)
            self._later(0, self.offline_reply, turn, pcm)

    async def _filler(self, turn):
        if self.state == St.THINKING and self.turn == turn and not self.got_audio:
            if self.player.play_filler() and self.tl:
                self.tl.set(filler=True)

    async def _cloud_timeout(self, turn, pcm):
        if self.turn == turn and self.state == St.THINKING and not self.got_audio:
            log.warning("turn %d: no brain audio after %.1fs, answering locally", turn, self.cfg.offline_after_s)
            await self.cloud.send("interrupt", turn=turn, played_ms=0)
            await self.offline_reply(turn, pcm)

    async def offline_reply(self, turn, pcm):
        self.cloud_turn = False
        self._set(St.OFFLINE_REPLY)
        if self.tl:
            self.tl.set(offline=True)
        loop = asyncio.get_running_loop()
        asr, tts = self.sp.get("asr"), self.sp.get("tts")
        text = ""
        if asr is not None and len(pcm):
            text = await loop.run_in_executor(None, asr.transcribe, pcm)
        if self.tl:
            self.tl.mark("stt_final_rx")
            self.tl.set(text=text)
        reply = await self.offline(text, {"turn": turn}) if text else None
        if self.turn != turn:
            return
        if reply and tts is not None:
            if self.tl:
                self.tl.mark("first_llm_delta_rx")
            pcm24 = await loop.run_in_executor(None, tts.synth, reply)
            if self.turn != turn:
                return
            if self.tl:
                self.tl.mark("first_tts_rx")
            self._set(St.SPEAKING)
            self.player.feed(pcm24)
            await self.player.wait_done()
            self.last_reply_s = self.player.played_ms / 1000.0
        if self.turn == turn and self.state in (St.SPEAKING, St.OFFLINE_REPLY):
            await self.start_listening("follow_up")

    async def barge_in(self):
        played = self.player.stop()
        turn = self.turn
        log.info("barge-in at %d ms", played)
        if self.tl:
            self.tl.set(barge_in_ms=played)
        if self.cloud_turn:
            await self.cloud.send("interrupt", turn=turn, played_ms=played)
        self._emit("barge_in", played_ms=played)
        await self.start_listening("barge_in")
        self.speech_active, self.speech_since = True, time.time()    # the user is already talking
        self._emit("speech_start")

    async def go_idle(self, reason):
        self._cancel_timers()
        if self.tl:
            self.tl.flush()
        self.speech_active, self.utt = False, []
        self._set(St.IDLE)
        self._emit("idle", reason=reason)

    # ------------------------------------------------------------------ brain callbacks
    async def on_tts_chunk(self, turn, pcm, final):
        if turn != self.turn or not self.cloud_turn:
            return                        # stale turn after barge-in / local fallback
        if pcm:
            if not self.got_audio:
                self.got_audio = True
                if self.tl:
                    self.tl.mark("first_tts_rx")
                if self.player.speaking:  # cut the filler, the real answer is here
                    self.player.stop()
            if self.state == St.THINKING:
                self._set(St.SPEAKING)
            self.player.feed(pcm)
        if final:                         # never block the link's frame loop while the reply plays out
            spawn(self._finish_turn(turn))

    def on_partial(self, turn, text):
        """Brain STT partial: an open-ended start ("tell me about...") gets the long endpoint (§8.6 #2)."""
        if turn == self.turn and self.state == St.LISTENING and text and LONG_TURN.match(text):
            self.endpoint_s = max(self.endpoint_s, self.cfg.endpoint_long_s)

    async def _finish_turn(self, turn):
        await self.player.wait_done()
        self.last_reply_s = self.player.played_ms / 1000.0
        if self.turn == turn and self.state in (St.SPEAKING, St.THINKING):
            await self.start_listening("follow_up")

    async def proactive(self, behaviour, info, fallback_text):
        """Robot-initiated turn: the brain phrases it (turn.begin with context.proactive, no audio); local template
        through Piper if the brain is offline or silent."""
        if self.busy:
            return False
        if self.cloud.online.is_set():
            self._cancel_timers()
            self.turn += 1
            turn = self.turn
            ctx = await self.context()
            if self.busy or self.turn != turn:          # woken while we gathered context: the person goes first
                return False
            ctx["proactive"] = dict(info, behaviour=behaviour)
            if (await self.cloud.send("turn.begin", turn=turn, context=ctx)
                    and await self.cloud.send("audio.end", turn=turn, proactive=True)):
                self.cloud_turn, self.got_audio = True, False
                if self.tl:
                    self.tl.begin(turn, reason="proactive")
                self._set(St.THINKING)
                self._later(self.cfg.offline_after_s + 2.0, self._proactive_timeout, turn, fallback_text)
                return True
        return await self.say(fallback_text)

    async def _proactive_timeout(self, turn, text):
        if self.turn == turn and self.state == St.THINKING and not self.got_audio:
            self.cloud_turn = False
            await self.go_idle("brain_silent")
            await self.say(text)

    async def say(self, text=None, pcm24=None, source="proactive"):
        """Speak outside a user turn (proactive, reminders). Returns False if a conversation is in progress."""
        if self.busy:
            return False
        if pcm24 is None and text:
            tts = self.sp.get("tts")
            if tts is None:
                return False
            pcm24 = await asyncio.get_running_loop().run_in_executor(None, tts.synth, text)
        if not pcm24 or self.busy:                  # woken during synthesis: the person goes first
            return False
        self.turn += 1
        turn = self.turn
        self.cloud_turn = False
        self._set(St.SPEAKING)
        self.player.feed(pcm24)
        await self.player.wait_done()
        if self.turn == turn and self.state == St.SPEAKING:     # not after a barge-in, which already listens
            await self.start_listening("follow_up")     # let the person answer without the wake word
        return True
