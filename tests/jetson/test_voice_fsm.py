"""Voice FSM endpointing (§8.4 / §8.6 #2) with a scripted VAD and a fake clock."""
import asyncio

import numpy as np

from beni_agent.config import Config
from beni_agent.voice import fsm as F


class FakeVad:
    speaking = False

    def accept(self, pcm):
        return self.speaking

    def reset(self):
        pass


class FakePlayer:
    speaking, started_at, played_ms = False, 0.0, 0


class FakeCloud:
    def __init__(self):
        self.online = asyncio.Event()

    async def send(self, *a, **kw):
        return False


def test_adaptive_endpoint(monkeypatch):
    clock = [100.0]
    monkeypatch.setattr(F.time, "time", lambda: clock[0])
    vad = FakeVad()

    async def offline(text, ctx):
        return None

    async def context():
        return {}

    async def go():
        v = F.VoiceLoop(Config(), {"vad": vad}, FakePlayer(), FakeCloud(), offline, context)
        pcm = np.zeros(160, np.float32)

        async def feed(t, speaking):
            clock[0], vad.speaking = 100.0 + t, speaking
            await v.on_audio(pcm, b"")

        async def utterance(quiet_at, wait):
            """Speech until quiet_at, then silence; returns whether the turn is still open `wait` s later."""
            turn = v.turn
            await feed(quiet_at - 0.5, True)
            await feed(quiet_at, False)
            await feed(quiet_at + wait, False)
            return v.turn == turn and v.state == F.St.LISTENING

        await v.start_listening("wake")
        assert v.endpoint_s == 0.6
        assert await utterance(1.0, 0.1)               # 0.45 VAD + 0.1 < 0.6
        await feed(1.2, False)                         # 0.45 + 0.2 >= 0.6 -> ended (offline: straight to follow-up)
        await asyncio.sleep(0.01)                      # the offline reply runs in the background
        assert v.state == F.St.LISTENING and v.endpoint_s == 0.6   # no reply was played -> not a quick exchange

        v.last_reply_s = 2.0
        await v.start_listening("follow_up")
        assert v.endpoint_s == 0.45 and not await utterance(3.0, 0.0)

        await v.start_listening("wake")
        v.on_partial(v.turn, "Tell me about the moon")
        assert v.endpoint_s == 0.8 and await utterance(5.0, 0.3)
        turn = v.turn
        await feed(5.4, False)                         # 0.45 + 0.4 >= 0.8 -> ended
        await asyncio.sleep(0.01)
        assert v.turn == turn + 1

    asyncio.run(go())


def test_timer_ending_in_follow_up_is_not_self_cancelled():
    """A cloud timeout answers locally, then opens the follow-up turn: turn.begin must still go out."""
    sent = []

    class Cloud(FakeCloud):
        async def send(self, type_, **kw):
            await asyncio.sleep(0)                     # a real socket write yields
            sent.append(type_)
            return True

    async def offline(text, ctx):
        return None

    async def context():
        await asyncio.sleep(0)
        return {}

    async def go():
        cloud = Cloud()
        cloud.online.set()
        cfg = Config()
        cfg.offline_after_s, cfg.filler_after_s = 0.01, 5.0
        v = F.VoiceLoop(cfg, {"vad": FakeVad()}, FakePlayer(), cloud, offline, context)
        await v.start_listening("wake")
        v.utt = [np.zeros(160, np.float32)]
        await v.end_of_speech()                        # brain never answers -> _cloud_timeout -> offline_reply
        await asyncio.sleep(0.1)
        assert v.state == F.St.LISTENING and v.cloud_turn
        assert sent[-1] == "turn.begin" and sent.count("turn.begin") == 2

    asyncio.run(go())


def test_say_yields_to_a_wake_during_synthesis():
    """A reminder being synthesized must not take over a turn the person just started with the wake word."""
    class Tts:
        def synth(self, text):
            import time
            time.sleep(0.05)
            return b"\0\0" * 240

    class Player(FakePlayer):
        def feed(self, pcm):
            raise AssertionError("must not speak over the person")

    async def offline(text, ctx):
        return None

    async def context():
        return {}

    async def go():
        v = F.VoiceLoop(Config(), {"vad": FakeVad(), "tts": Tts()}, Player(), FakeCloud(), offline, context)
        say = asyncio.ensure_future(v.say("time for your medicine"))
        await asyncio.sleep(0.01)
        await v.start_listening("wake")
        turn = v.turn
        assert await say is False
        assert v.state == F.St.LISTENING and v.turn == turn

    asyncio.run(go())


def test_say_after_barge_in_keeps_the_new_turn():
    class Player(FakePlayer):
        done = None

        def feed(self, pcm):
            self.done = asyncio.Event()

        async def wait_done(self):
            await self.done.wait()

        def stop(self):
            self.done.set()
            return 300

    async def offline(text, ctx):
        return None

    async def context():
        return {}

    async def go():
        p = Player()
        v = F.VoiceLoop(Config(), {"vad": FakeVad()}, p, FakeCloud(), offline, context)
        say = asyncio.ensure_future(v.say(pcm24=b"\0\0" * 240))
        await asyncio.sleep(0)
        await v.barge_in()
        assert await say is True
        assert v.state == F.St.LISTENING and v.speech_active      # still mid-utterance, not reset by say()

    asyncio.run(go())
