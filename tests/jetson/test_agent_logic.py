"""Pure-logic parts of the Jetson agent: reminders, bandits, admission, modes, offline intents, tegrastats."""
import datetime as dt
import random

import pytest

from beni_agent.behaviour.bandits import CAP, Bandits
from beni_agent.behaviour.offline import match_intent
from beni_agent.behaviour.proactive import parse_when
from beni_agent.scheduler import admit as A
from beni_agent.scheduler import thermal as T
from beni_agent.scheduler.modes import decide, empty_check, want_llm
from beni_common import tegrastats

NOW = dt.datetime(2026, 9, 27, 12, 0).timestamp()


@pytest.mark.parametrize("s, expect", [
    ("in 20 minutes", NOW + 1200), ("in 2 hours", NOW + 7200), ("18:30", NOW + 6.5 * 3600),
    ("09:00", NOW + 21 * 3600), ("2026-09-28T08:15", dt.datetime(2026, 9, 28, 8, 15).timestamp()),
    ("whenever", None), ("", None)])
def test_parse_when(s, expect):
    assert parse_when(s, NOW) == expect


def test_bandits_learn_and_cap():
    kv = {}
    b = Bandits(lambda k, d=None: kv.get(k, d), kv.__setitem__, rng=random.Random(1))
    for _ in range(60):
        b.update("greet", 1, "p1", t=NOW)
        b.update("chat", -1, "p1", t=NOW)
    a, bb = b.params("greet", "p1", t=NOW)
    assert a + bb <= CAP + 1.0 and a > bb                         # soft cap: rescale then floor at the prior
    wins = sum(b.choose(["greet", "chat"], "p1", t=NOW)[0] == "greet" for _ in range(100))
    assert wins > 90
    assert "bandit.p1" in kv                                    # persisted per person


def _stats(**kw):
    s = A.Stats(clock=lambda: 100.0)
    s.add({"gpu": kw.pop("gpu", 10), "cpu_avg": kw.pop("cpu", 10), "nvdec": 0, "temp_gpu": kw.pop("temp", 50)}, t=99.5)
    for k, v in kw.items():
        setattr(s, k, v)
    return s


def test_admission():
    d = A.admit(_stats())
    assert not d["pause"] and A.job_allowed(d, ["gpu", "ram"])
    assert not A.job_allowed(A.admit(_stats(gpu=70)), ["gpu"])
    assert A.job_allowed(A.admit(_stats(gpu=70)), ["cpu"])
    assert A.admit(_stats(voice_active=True))["pause"]
    assert not A.job_allowed(A.admit(_stats(batt_pct=30)), ["power"])
    assert A.job_allowed(A.admit(_stats(batt_pct=30, on_charger=True)), ["power"])


def test_modes():
    assert decide(5, False, False, 0, True) == "critical"
    assert decide(15, False, False, 0, False) == "low"
    assert decide(15, True, True, 0, False) == "idle"
    assert decide(25, False, False, 0, False, power_w=9.0) == "low"      # §4 #47: INA3221 draw moves "low" up
    assert decide(25, False, False, 0, False, power_w=5.0) == "active"
    assert decide(25, True, False, 0, False, power_w=9.0) == "active"
    assert decide(80, False, False, 0, True) == "active"
    assert decide(80, False, False, 3600, False) == "idle"
    assert want_llm(False, 120, 900, False) and not want_llm(False, 120, 300, False)
    assert not want_llm(True, 400, 900, True) and want_llm(True, 10, 900, True)
    assert decide(80, False, False, 0, False, hot=True) == "idle"
    assert decide(80, True, True, 3600, False, night=True) == "replay"
    assert decide(80, True, True, 60, False, night=True) == "idle"
    assert decide(80, False, False, 3600, False, night=True) == "idle"
    assert decide(80, False, False, 0, True, hot=True) == "active"


def test_empty_battery_power_off():
    """§15.1 critical: off at < 5 % on battery, only after 60 s (a one-sample blip or docking cancels it)."""
    assert empty_check(None, 4, False, 100.0) == (100.0, False)
    assert empty_check(100.0, 4, False, 130.0) == (100.0, False)
    assert empty_check(100.0, 6, False, 130.0) == (None, False)
    assert empty_check(100.0, 4, True, 200.0) == (None, False)
    assert empty_check(None, None, False, 200.0) == (None, False)
    assert empty_check(100.0, 3, False, 160.0) == (100.0, True)


def test_applier_pauses_cam0(monkeypatch):
    """§15.1 idle-watch: CAM0's Argus session stops in idle/replay and restarts on the way out."""
    import asyncio
    from beni_agent.scheduler import modes as M

    async def run(*cmd):
        return True
    monkeypatch.setattr(M, "_run", run)
    sent, ok = [], [True]

    async def request(ep, msg):
        sent.append(msg)
        return {"ok": ok[0] or msg["op"] != "cam"}
    a = M.Applier(request, clock=lambda: 0.0)

    async def go(*modes):
        for m in modes:
            await a.set_mode(m, force=True)
    asyncio.run(go("active", "idle", "replay", "active"))
    assert [m["on"] for m in sent if m["op"] == "cam"] == [0, 1]
    ok[0] = False   # DeepStream path: unknown op -> state unknown, re-sent on the next change
    asyncio.run(go("idle", "active"))
    assert [m["on"] for m in sent if m["op"] == "cam"] == [0, 1, 0, 1] and a.cam0 is None


def test_fan_and_thermal(tmp_path):
    assert T.fan_pwm(40) == T.FAN_MIN and T.fan_pwm(75) == T.FAN_MAX and T.fan_pwm(None) == T.FAN_MAX
    assert T.FAN_MIN < T.fan_pwm(62.5) < T.FAN_MAX
    assert T.next_pwm(None, 40) == T.FAN_MIN and T.next_pwm(100, 75) == T.FAN_MAX       # spin up at once
    assert T.next_pwm(255, 40) == 255 - T.FAN_DOWN_STEP                                  # spin down slowly
    (tmp_path / "target_pwm").write_text("0")
    (tmp_path / "temp_control").write_text("1")
    fan = T.Fan(str(tmp_path))
    assert (tmp_path / "temp_control").read_text() == "0"
    assert fan.set(70) == 255 and (tmp_path / "target_pwm").read_text() == "255"
    assert not T.Fan(str(tmp_path / "missing")).ok
    g = T.ThermalGuard()
    assert not g.feed(79) and g.feed(81) and g.feed(75) and not g.feed(72) and not g.feed(None)


@pytest.mark.parametrize("text, action", [
    ("stop!", "stop"), ("Beni come here", "come_here"), ("go to your dock", "goto_dock"),
    ("not now please", "quiet"), ("please forget me", "forget_request"), ("hello there", None),
    ("hey Beni, close your eyes", "camera_privacy"), ("turn off your cameras", "camera_privacy")])
def test_offline_intents(text, action):
    r = match_intent(text, {"names": ["Nimal"]})
    assert r is not None and r[0] == action


@pytest.mark.parametrize("text, action, args", [
    ("this is the kitchen", "save_place", {"name": "kitchen"}),
    ("remember this spot as Mom's room.", "save_place", {"name": "Mom's room"}),
    ("go to the living room please", "move_to", {"place": "living room"}),
    ("this is Mom's room now", "save_place", {"name": "Mom's room"}),
    ("go to bed", "goto_dock", {}), ("follow me", "follow_person", {"name": "me"}),
    ("privacy mode please", "camera_privacy", {"on": True}), ("privacy mode off", "camera_privacy", {"on": False}),
    ("you can open your eyes now", "camera_privacy", {"on": False}), ("switch your camera back on", "camera_privacy",
                                                                      {"on": False})])
def test_offline_place_intents(text, action, args):
    assert match_intent(text)[0::2] == (action, args)


def test_offline_cold_start_note():
    """§10.4: the first offline answer after a cold-start request says the big brain is waking up."""
    import asyncio
    from beni_agent.behaviour.offline import WAKING, OfflineBrain
    b = OfflineBrain(None)
    b.waking = True
    first, second = asyncio.run(b("stop")), asyncio.run(b("stop"))
    assert first.startswith(WAKING) and not second.startswith(WAKING) and not b.waking


def test_offline_where_is():
    r = match_intent("where are my keys?", {"find": lambda o: "found " + o})
    assert r == (None, "found keys", {})


def test_offline_intent_miss():
    assert match_intent("explain quantum physics") is None
    assert match_intent("this is great") is None and match_intent("this is my friend Ana") is None


@pytest.mark.parametrize("text, action", [
    ("hi, where are my keys", None), ("hello, go to the kitchen", "move_to"), ("hey what time is it", None)])
def test_offline_greeting_does_not_hide_a_request(text, action):
    r = match_intent(text, {"find": lambda o: "found " + o})
    assert r[0] == action and not r[1].startswith("Hello")


def test_tegrastats_parse():
    line = ("RAM 2519/3956MB (lfb 3x4MB) SWAP 0/1978MB CPU [12%@1479,off,7%@1479,20%@1479] EMC_FREQ 9%@1600 "
            "GR3D_FREQ 45%@921 NVDEC 716 APE 25 PLL@33C CPU@36.5C PMIC@100C GPU@34C AO@41C thermal@35.25C "
            "POM_5V_IN 3187/3187 POM_5V_GPU 450/450 POM_5V_CPU 612/612")
    r = tegrastats.parse(line, t=1.0)
    assert (r["ram_mb"], r["gpu"], r["cpu_max"], r["temp_gpu"], r["p_in_mw"]) == (2519, 45.0, 20, 34.0, 3187)
    assert r["cpu_avg"] == pytest.approx(9.75)


def test_lifecycle_renders_kernel_metadata(tmp_path):
    import json
    import os
    from types import SimpleNamespace
    from beni_agent.cloud.lifecycle import Lifecycle
    root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    cfg = SimpleNamespace(kaggle_kernel="alice/beni-brain", kaggle_dir=os.path.join(root, "kaggle"))
    d = Lifecycle(cfg, None, None)._render()
    meta = json.load(open(os.path.join(d, "kernel-metadata.json")))
    assert meta["id"] == "alice/beni-brain" and meta["dataset_sources"] == ["alice/beni-wheelhouse"]
    assert meta["machine_shape"] == "NvidiaTeslaT4" and sorted(os.listdir(d)) == ["brain_notebook.py",
                                                                                   "kernel-metadata.json"]


def test_gpio_debounce_and_setup(tmp_path):
    from beni_agent import gpio
    d = gpio.Debounce(True, 0.05)
    assert d.feed(False, 0.0) is None and d.feed(True, 0.01) is None       # bounce ignored
    assert d.feed(False, 0.02) is None and d.feed(False, 0.08) is False and d.level is False
    assert gpio.Debounce(False, 0.0).feed(True, 5.0) is True                # e-stop: no hold
    pin = tmp_path / "gpio14"
    pin.mkdir()
    for name, v in (("direction", "out"), ("edge", "none"), ("value", "0")):
        (pin / name).write_text(v)
    (tmp_path / "export").write_text("")
    assert gpio.setup(14, root=str(tmp_path)) == str(pin / "value")
    assert (pin / "direction").read_text() == "in" and (pin / "edge").read_text() == "both"


def test_phone_presence():
    from beni_agent.perception import presence as P
    mac = "AA:BB:CC:DD:EE:FF"
    assert P.parse_phones("Nimal=aa:bb:cc:dd:ee:ff, bad=xx") == {mac: "Nimal"}
    seen = []
    p = P.Presence({mac: "Nimal"}, lambda n, h: seen.append((n, h)))
    p.feed(mac, False)
    assert seen == [] and p.names_home() == []                        # first result "away": no event
    p.feed(mac, True)
    assert seen == [("Nimal", True)] and p.names_home() == ["Nimal"]
    for _ in range(P.MISSES_AWAY - 1):
        p.feed(mac, False)
    assert p.names_home() == ["Nimal"]                               # dozing radio is not "left"
    p.feed(mac, False)
    assert seen[-1] == ("Nimal", False)


def test_touch_gestures():
    from beni_agent import touch as T
    s = T.Stroke()
    s.down(0.0, 0.5, 0.5)
    s.move(0.51, 0.5)
    assert s.up(0.2) == "tap"
    s.down(0.0, 0.2, 0.5)
    for i in range(10):
        s.move(0.2 + 0.03 * i, 0.5)
    assert s.up(0.6) == "pet"
    s.down(0.0, 0.5, 0.5)
    assert s.up(1.0) == "hold" and s.up(1.1) is None
    assert T.parse_cal("100,3900,3900,100") == (100, 3900, 3900, 100) and T.parse_cal("x") is None
    assert T.norm(3900, 100, 3900) == 1.0 and T.norm(3900, 3900, 100) == 0.0 and T.norm(50, 100, 3900) == 0.0
    assert T.EVENT.size == 24 or T.EVENT.size == 16          # 16 on 32-bit hosts; the Nano is aarch64 (24)


def test_db_jobs_are_never_sigstopped(monkeypatch):
    """A job holding the memory DB's write lock runs to the end; other background jobs pause at once."""
    import signal
    from beni_agent.scheduler.__main__ import Job
    monkeypatch.setattr(signal, "SIGSTOP", 19, raising=False)       # absent on Windows dev boxes
    monkeypatch.setattr(signal, "SIGCONT", 18, raising=False)

    class Proc:
        returncode = None

        def __init__(self):
            self.sigs = []

        def poll(self):
            return None

        def send_signal(self, s):
            self.sigs.append(s)

    db, logs = Job("fts", [], ("cpu",), 60, pausable=False), Job("gzip", [], ("cpu",), 60)
    for j in (db, logs):
        j.proc = Proc()
        j.step(False, 0.0)
    assert db.proc.sigs == [] and not db.stopped
    assert len(logs.proc.sigs) == 1 and logs.stopped
