"""beni_agent entry point: wires audio, voice FSM, cloud link, memory sync, perception and behaviours on one
asyncio loop (cores 2-3, py3.8). Heavy CPU work (ASR/TTS/embeddings/backup) goes to the default executor.
"""
import asyncio
import logging
import os
import signal
import time
from concurrent.futures import ThreadPoolExecutor

from beni_common import schemas as S
from beni_common.memory import MemoryStore
from beni_common.memory import embed as embed_onnx

from . import bus as B
from . import gpio
from .touch import Touch, parse_cal
from .actions import Actions
from .behaviour.bandits import Bandits
from .behaviour.offline import OfflineBrain
from .behaviour.proactive import Proactive
from .cloud.lifecycle import Lifecycle
from .cloud.link import CloudLink
from .config import Config, in_windows
from .memory import backup
from .memory.sync import MemorySync
from .perception.gestures import Gestures
from .perception.identity import FaceIdentifier, Gallery
from .perception.places import Places
from .perception.presence import Presence, parse_phones
from .perception.replay import Replay
from .perception.thumbs import Thumbs
from .perception.world import World
from .util import TurnLog, sd_notify, setup_logging, spawn
from .voice import speech
from .voice.audio import MicReceiver, Player
from .voice.fsm import VoiceLoop

log = logging.getLogger("agent")


class Agent:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.store = MemoryStore(cfg.db, node="jetson")
        self.bus = B.Bus("agent")
        self.events = self.bus.pub(S.EP["events"])
        self.embed = embed_onnx.load(cfg.models)
        self.gallery = Gallery(self.store, "face_exemplar")
        self.ident = FaceIdentifier(self.gallery)
        self.places = Places(self.store)
        self.world = World(self.store, self.ident, self.events, places=self.places)
        self.tl = TurnLog(os.path.join(cfg.logs, "turns.jsonl"))
        self.link = CloudLink(cfg.brain_urls, cfg.token, self.on_frame, cfg.robot_id,
                              memory_rev=lambda: self.store.peer_cursor("brain"), on_state=self.on_link_state)
        self.sync = MemorySync(self.store, self.link, on_applied=self.on_applied)
        self.lifecycle = Lifecycle(cfg, self.store.kv_get, self.store.kv_set, self.link)
        self.sp = speech.load(cfg.models)
        self.player = Player(cfg.tts_port, fillers_dir=cfg.m("fillers"))
        self.player.on_first_audio = lambda t: self.tl.mark("first_audio_out")
        self.actions = Actions(self.bus, self.world, self.ident, None, self.store, self.sync, self.link, self.places)
        self.offline = OfflineBrain(cfg, self.world, self.actions, self.store, self.embed)
        self.voice = VoiceLoop(cfg, self.sp, self.player, self.link, self.offline, self.context, self.events,
                               self.tl, on_wake=self.on_wake)
        self.bandits = Bandits(self.store.kv_get, self.store.kv_set)
        self.proactive = Proactive(cfg, self.world, self.voice, self.bandits, self.store, self.actions)
        self.actions.proactive = self.proactive
        self.actions.emit = self.events.send
        self.estop = False
        self.touch = None
        if cfg.touch != "off":
            self.touch = Touch(self.on_touch, irq_value=True if cfg.gpio_penirq >= 0 else None,
                               cal=parse_cal(cfg.touch_cal), evdev=cfg.touch if cfg.touch.startswith("/") else None)
        self.presence = Presence(parse_phones(cfg.phones), self.on_phone)
        self.world.phones_home = self.presence.names_home
        self.thumbs = Thumbs(self.store, self.actions.snapshot, os.path.join(os.path.dirname(cfg.db), "thumbs"))
        self.ident.on_exemplar = lambda eid, cam, bbox: spawn(self.thumbs.face(eid, cam, bbox))
        self.world.on_sighting = self.on_sighting
        self.gestures = Gestures(self.bus.request, self.on_gesture)
        self.replay = Replay(self.store, self.gallery, self.bus.request, cfg.rec_dir, self.thumbs.root, self.link)

    # ------------------------------------------------------------------ callbacks
    async def context(self):
        return self.world.context()

    def on_wake(self):
        self.gestures.watch_soon(0, 20)                          # "what's that?" + pointing, during the turn
        if not self.link.online.is_set() and self.lifecycle.ensure_running():
            log.info("wake while brain is down: cold start requested")
            self.offline.waking = True

    def on_link_state(self, online, hello):
        self.events.send(b"event", name="brain", online=online)
        if online:
            spawn(self.sync.on_hello(hello))

    def on_applied(self, tables):
        if tables & {"face_exemplar", "person"}:
            self.gallery.reload()
            self.world.invalidate_names()
        if "place" in tables:
            self.places.invalidate()

    async def on_frame(self, f):
        t, turn = f.get("type"), f.get("turn", 0)
        if t == "tts.chunk":
            if f.get("emo"):
                await self.actions("set_expression", {"expression": f["emo"]})
            await self.voice.on_tts_chunk(turn, f.get("pcm16_24k") or b"", bool(f.get("final")))
        elif t == "stt.partial":
            self.voice.on_partial(turn, f.get("text") or "")
        elif t == "llm.delta":
            self.tl.mark("first_llm_delta_rx", turn)
        elif t == "stt.final":
            self.tl.mark("stt_final_rx", turn)
            self.tl.set(text=f.get("text"))
            self.proactive.on_user_text(f.get("text"))
            self.events.send(b"event", name="stt_final", turn=turn, text=f.get("text"), speaker=f.get("speaker"))
        elif t in ("action", "vision.request"):
            spawn(self.actions.on_frame(f))      # never block the frame loop on a 20 s action
        elif t in ("memory.delta", "memory.ack"):
            await self.sync.on_frame(f)
        elif t == "turn.end":
            await self.voice.on_tts_chunk(turn, b"", True)       # no-audio turns (pure actions) still end cleanly
        elif t == "replay.result":
            self.replay.on_result(f)
        elif t == "scene":
            self.world.last_scene_caption = f.get("caption", "")
        elif t == "error":
            log.warning("brain error: %s", f.get("error"))

    def on_robot(self, topic, msg):
        self.world.on_robot_state(topic, msg)
        if "mic_muted" in msg:
            self.voice.muted = bool(msg["mic_muted"])

    def on_gpio(self, name, level):
        """§13.4: the ESP32 holds pin 13 high while any safety stop is active; the mute button pulls low."""
        if name == "estop":
            if level != self.estop:
                log.warning("e-stop line %s", "ACTIVE" if level else "clear")
            self.estop = self.world.estop = level
            self.events.send(b"event", name="estop", active=level)
        elif name == "penirq" and self.touch is not None:
            self.touch.pen_irq(level)
        elif name == "mute" and not level:                       # press toggles
            self.voice.muted = not self.voice.muted
            self.events.send(b"event", name="mic_muted", muted=self.voice.muted)

    def on_touch(self, gesture, x, y):
        """§3.11: tap the face to talk (no wake word needed), stroke it to pet Beni."""
        self.events.send(b"event", name="touch", gesture=gesture, x=x, y=y)
        if gesture == "tap" and not self.voice.busy and not self.voice.muted:
            spawn(self.voice.start_listening("wake"))
        elif gesture == "pet":
            self.world.note("someone petted my face")
            spawn(self.actions("set_expression", {"expression": "happy"}))

    def on_phone(self, name, home):
        """§4 item 56: an owner's phone came home -> pre-warm the brain (a Kaggle cold start takes minutes)."""
        self.world.note("%s's phone %s" % (name, "came home" if home else "left"))
        self.events.send(b"event", name="phone_presence", person=name, home=home)
        lt = time.localtime()
        if home and not in_windows(self.cfg.quiet_hours, lt.tm_hour * 60 + lt.tm_min):
            self.lifecycle.ensure_running(20)

    def on_vision(self, topic, msg):
        if topic == S.T_POSE:
            self.gestures.on_msg(topic, msg)
            return
        seen = self.world.persons_in_view
        self.world.on_vision(topic, msg)
        self.actions.on_vision(topic, msg)
        if self.world.persons_in_view and not seen:
            self.gestures.watch_soon(0, 10)                      # someone came into view: a wave hello?

    def on_sighting(self, pid, name, cam, place):
        spawn(self.thumbs.sighting(pid, name, cam, place))
        self.gestures.watch_soon(cam, 10)

    def on_gesture(self, name, cam, tid, extra):
        """§4 item 39: a wave starts a conversation like a tap on the face; a point goes into the turn context."""
        tr = self.ident.tracks.get((cam, tid))
        pid = tr.person if tr is not None else None
        who = self.world.name(pid) if pid else "someone"
        self.events.send(b"event", name="gesture", gesture=name, cam=cam, tid=tid, person=pid, **extra)
        if name == "wave":
            self.world.note("%s waved at me" % who)
            spawn(self.actions("set_expression", {"expression": "happy"}))
            if not self.voice.busy and not self.voice.muted:
                spawn(self.voice.start_listening("wake"))
        elif name == "point":
            self.world.note("%s pointed to my %s" % (who, extra.get("dir")))

    # ------------------------------------------------------------------ run
    async def watchdog(self):
        sd_notify("READY=1")
        while True:
            sd_notify("WATCHDOG=1")
            sd_notify("STATUS=%s brain=%s rtt=%s" % (self.voice.state.name, self.link.online.is_set(),
                                                      int(self.link.rtt_ms or 0)))
            await asyncio.sleep(10)

    async def run(self):
        mic = await MicReceiver.open(self.cfg.mic_port)
        vis = self.bus.sub(S.EP["vision"], S.T_DET, S.T_JPEG, S.T_POSE)
        rob = self.bus.sub(S.EP["robot_state"], S.T_STATE)
        rep = self.bus.sub(S.EP["vision"], S.T_REPLAY)
        sch = self.bus.sub(S.EP["sched"], b"sched")
        tasks = [
            self.voice.run(mic), self.player.run(), self.link.run(), self.sync.run(), self.lifecycle.run(),
            self.proactive.run(), self.watchdog(), self.presence.run(), self.thumbs.run(),
            B.pump(vis, self.on_vision, "vision"), B.pump(rob, self.on_robot, "robot"), self.replay.run(),
            B.pump(rep, self.replay.on_msg, "replay"), B.pump(sch, self.replay.on_sched, "sched"),
            backup.nightly(self.store, self.cfg, busy=lambda: self.voice.busy),
        ]
        pins = gpio.Watcher(asyncio.get_running_loop(), self.on_gpio)
        if self.cfg.gpio_estop >= 0:
            pins.add("estop", gpio.setup(self.cfg.gpio_estop))
        if self.cfg.gpio_mute >= 0:
            pins.add("mute", gpio.setup(self.cfg.gpio_mute), hold_s=0.05)
        if self.touch is not None:
            if self.cfg.gpio_penirq >= 0:
                pins.add("penirq", gpio.setup(self.cfg.gpio_penirq))
            spawn(self.touch.run())
        pins.start()
        spawn(self.actions.restore_privacy())
        log.info("agent up: models=%s brain=%s", sorted(self.sp), self.cfg.brain_urls)
        await asyncio.gather(*tasks)


def main():
    setup_logging()
    cfg = Config.from_env()
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    loop.set_default_executor(ThreadPoolExecutor(2))            # ASR/TTS/embeddings; 2 = cores 2-3
    agent = Agent(cfg)
    task = loop.create_task(agent.run())
    for sig in (signal.SIGTERM, signal.SIGINT):
        try:
            loop.add_signal_handler(sig, task.cancel)
        except NotImplementedError:
            pass
    try:
        loop.run_until_complete(task)
    except asyncio.CancelledError:
        pass
    finally:
        agent.tl.flush()
        agent.store.close()
    return 0
