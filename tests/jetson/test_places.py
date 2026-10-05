"""§11.10 spatial memory: place resolution, where-am-I, sightings -> beliefs, nav experience, tool -> bridge ops."""
import asyncio
import json
from types import SimpleNamespace

import pytest

from beni_agent.actions import Actions
from beni_agent.perception.places import Places, norm
from beni_common import schemas as S
from beni_common.memory import MemoryStore


@pytest.fixture
def places(tmp_path):
    st = MemoryStore(str(tmp_path / "j.db"), node="jetson")
    now = [1000.0]
    p = Places(st, clock=lambda: now[0])
    p.now = now
    p.save("Living room", {"x": 0.0, "y": 0.0, "yaw": 0.0})
    p.save("kitchen", {"x": 4.0, "y": 0.0, "yaw": 1.57})
    p.save("charger", {"x": -1.0, "y": 2.0, "yaw": 3.14})
    yield p
    st.close()


def test_norm_and_resolve(places):
    assert norm("the Living Room") == "living" and norm("Mom's room") == "mom"
    assert places.resolve("living room")["name"] == "Living room"
    assert places.resolve("the kitchen")["name"] == "kitchen"
    assert places.resolve("dock")["kind"] == "dock" and places.dock()["name"] == "charger"
    assert places.resolve("garage") is None and places.resolve("") is None


def test_save_moves_existing(places):
    n = len(places.rows())
    places.save("Kitchen", {"x": 5.0, "y": 1.0, "yaw": 0.0})
    assert len(places.rows()) == n and places.resolve("kitchen")["x"] == 5.0
    places.save("dock", {"x": -2.0, "y": 2.0, "yaw": 0.0})       # the dock is unique: moves the charger row
    assert len(places.rows()) == n and places.dock()["x"] == -2.0


def test_where_am_i_hysteresis(places):
    entered, _ = places.on_state({"pose": {"x": 3.8, "y": 0.1, "yaw": 0}})
    assert entered["name"] == "kitchen"
    assert places.on_state({"pose": {"x": 3.9, "y": 0.0, "yaw": 0}})[0] is None     # no re-entry spam
    places.on_state({"pose": {"x": 3.0, "y": 0.0, "yaw": 0}})                       # just outside: keep kitchen
    assert places.current["name"] == "kitchen"
    assert places.on_state({"pose": {"x": 0.1, "y": 0.0, "yaw": 0}})[0]["name"] == "Living room"


def test_sightings_update_beliefs(places):
    cup = S.COCO.index("cup")
    places.on_state({"pose": {"x": 4.0, "y": 0.0, "yaw": 0}})
    objs = [{"cam": 1, "tid": 3, "cls": cup, "conf": 0.8, "x": 4.5, "y": 0.2},
            {"cam": 1, "tid": 4, "cls": 0, "conf": 0.9, "x": 4.2, "y": 0.0},           # person: skipped
            {"cam": 1, "tid": 5, "cls": S.COCO.index("book"), "conf": 0.3, "x": 4.2, "y": 0.0}]   # low conf
    for _ in range(5):
        places.on_state({"pose": {"x": 4.0, "y": 0.0, "yaw": 0}, "objs": objs})
    st = places.store
    assert [r["label"] for r in st.q("SELECT label FROM object_sighting")] == ["cup"]    # rate-limited
    places.now[0] += 200
    places.on_objects(objs)
    b = st.get("object_belief", "obj-cup")
    kitchen = places.resolve("kitchen")["id"]
    assert b["last_place_id"] == kitchen and json.loads(b["place_hist"]) == {kitchen: 2}
    assert len(st.q("SELECT id FROM object_sighting")) == 2


def test_nav_experience_logged_once(places):
    places.on_state({"pose": {"x": 0.0, "y": 0.0, "yaw": 0}})
    run = {"id": 7, "op": "goto", "status": "running", "detail": "kitchen", "dur": 1.0, "recov": 0, "stuck": []}
    assert places.on_state({"pose": {"x": 1.0, "y": 0.0, "yaw": 0}, "task": run})[1] is None
    done = dict(run, status="failed", detail="could not reach kitchen", dur=42.0, recov=2, stuck=[[2.0, 0.5]])
    row = places.on_state({"pose": {"x": 2.0, "y": 0.5, "yaw": 0}, "task": done})[1]
    assert row["from_place"] == "Living room" and row["to_place"] == "kitchen" and row["success"] == 0
    assert json.loads(row["stuck_xy"]) == [[2.0, 0.5]] and row["recoveries"] == 2
    assert places.on_state({"pose": {"x": 2.0, "y": 0.5, "yaw": 0}, "task": done})[1] is None
    look = {"id": 8, "op": "look_at", "status": "succeeded"}
    assert places.on_state({"pose": {"x": 2.0, "y": 0.5, "yaw": 0}, "task": look})[1] is None
    assert len(places.store.q("SELECT id FROM nav_experience")) == 1


class FakeBus:
    def __init__(self):
        self.sent = []

    def push(self, ep):
        return SimpleNamespace(send=lambda b: None)

    async def request(self, ep, msg, timeout=3.0):
        self.sent.append(msg)
        if msg["op"] == "save_place":
            return {"ok": True, "result": {"x": 1.5, "y": -2.0, "yaw": 0.3, "map_id": "home", "frame": "map"}}
        return {"ok": True, "result": "going"}


def test_actions_resolve_places_and_people(places):
    st = places.store
    pid = st.put("person", {"display_name": "Nimal"})
    ident = SimpleNamespace(tracks={(0, 12): SimpleNamespace(person=pid, last=5.0, key=(0, 12)),
                                    (1, 3): SimpleNamespace(person=pid, last=9.0, key=(1, 3))})
    bus = FakeBus()
    a = Actions(bus, identifier=ident, store=st, places=places)
    run = asyncio.new_event_loop().run_until_complete
    assert run(a("move_to", {"place": "the kitchen"}))["ok"]
    assert bus.sent[-1] == {"op": "goto", "place": "kitchen", "x": 4.0, "y": 0.0, "yaw": 1.57}
    r = run(a("move_to", {"place": "garage"}))
    assert not r["ok"] and "garage" in r["result"] and len(bus.sent) == 1        # never reaches the bridge
    run(a("goto_dock", {}))
    assert bus.sent[-1]["op"] == "dock" and bus.sent[-1]["x"] == -1.0
    run(a("follow_person", {"name": "nimal"}))
    assert bus.sent[-1] == {"op": "follow", "name": "nimal", "cam": 1, "tid": 3}   # freshest track wins
    run(a("look_at", {"target": "me"}))
    assert "tid" not in bus.sent[-1]
    run(a("search_room", {"object": "cup"}))
    assert bus.sent[-1] == {"op": "search", "object": "cup"}
    r = run(a("save_place", {"name": "hallway"}))
    assert r["ok"] and places.resolve("hallway")["x"] == 1.5


def test_keepout_mask(places, tmp_path):
    from beni_agent.memory import keepout
    rnd = __import__("random").Random(3)
    st, pts = places.store, [(2.0 + rnd.uniform(-.1, .1), 1.0 + rnd.uniform(-.1, .1)) for _ in range(6)]
    for i in range(0, 6, 2):
        st.put("nav_experience", {"t": 1e12, "success": 0, "stuck_xy": json.dumps(pts[i:i + 2])})
    st.put("nav_experience", {"t": 1e12, "success": 0, "stuck_xy": "[[9, 9]]"})        # lone point: noise
    places.save("Mom's room", {"x": -3.0, "y": 0.0}, kind="keepout")
    zs = keepout.zones(st, now=1e12)
    assert len(zs) == 2 and sorted(z[3] for z in zs) == [keepout.HIGH, 100]
    assert abs(zs[0][0] - 2.0) < 0.1 and abs(zs[0][1] - 1.0) < 0.1
    yml = keepout.write_mask(zs, str(tmp_path))
    meta = open(yml).read()
    assert "mode: scale" in meta and "image: keepout.pgm" in meta
    raw = open(str(tmp_path / "keepout.pgm"), "rb").read()
    hdr = raw.split(b"\n", 3)
    w, h = map(int, hdr[1].split())
    px = hdr[3]
    assert len(px) == w * h and min(px) == 0 and max(px) == 255      # lethal core, free elsewhere
    ox, oy = [float(v) for v in meta.split("origin: [")[1].split(",")[:2]]
    col, row = int((2.0 - ox) / keepout.RES), h - 1 - int((1.0 - oy) / keepout.RES)
    assert 60 < px[row * w + col] < 120                                # graded cluster centre (~70 %)
    assert keepout.write_mask([], str(tmp_path)) is None and not (tmp_path / "keepout.pgm").exists()


def test_camera_privacy(tmp_path):
    """§12.3: privacy stops the running vision unit, closes the eyes, survives a restart, and restarts that unit."""
    st = MemoryStore(str(tmp_path / "m.db"))
    world, cmds, events = SimpleNamespace(privacy=False), [], []
    active = {"beni-vision": "active"}

    async def cmd(*argv):
        cmds.append(argv)
        return (0, active.get(argv[-1], "inactive")) if argv[1] == "is-active" else (0, "")
    a = Actions(FakeBus(), world=world, store=st)
    a.cmd, a.emit = cmd, lambda topic, **f: events.append(f)
    run = asyncio.new_event_loop().run_until_complete
    assert run(a("camera_privacy", {"on": True}))["ok"]
    assert ("sudo", "-n", "/bin/systemctl", "stop", "beni-vision") in cmds and world.privacy
    assert events[-1] == {"name": "privacy", "on": 1} and st.kv_get("privacy") == {"on": True, "units": ["beni-vision"]}
    active.clear()
    run(a.restore_privacy())                                 # nothing active any more: still remembers the unit
    assert st.kv_get("privacy")["units"] == ["beni-vision"]
    assert run(a("camera_privacy", {"on": "false"}))["ok"] and not world.privacy
    assert cmds[-1] == ("sudo", "-n", "/bin/systemctl", "start", "beni-vision") and events[-1]["on"] == 0
    cmds.clear()
    run(a.restore_privacy())
    assert not cmds


def test_estop_blocks_motion(places):
    """While the e-stop GPIO is held, motion requests never reach the bridge; "stop" still does."""
    bus, world = FakeBus(), SimpleNamespace(estop=True)
    a = Actions(bus, world=world, store=places.store, places=places)
    run = asyncio.new_event_loop().run_until_complete
    for op, args in (("move_to", {"place": "kitchen"}), ("goto_dock", {}), ("come_here", {})):
        r = run(a(op, args))
        assert not r["ok"] and "emergency" in r["result"]
    assert not bus.sent
    assert run(a("stop", {}))["ok"] and bus.sent[-1]["op"] == "stop"
    world.estop = False
    assert run(a("move_to", {"place": "kitchen"}))["ok"]
