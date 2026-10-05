"""Sleep replay (§11.9 step 9): segment picking, naming face tracks, stranger clusters across nights, brain results."""
import asyncio
import os
import time
from types import SimpleNamespace

import numpy as np

from beni_agent.behaviour.proactive import Proactive, when_phrase
from beni_agent.perception import replay as R
from beni_agent.perception.identity import Gallery
from beni_common import schemas as S
from beni_common.memory import MemoryStore, routines, to_blob


def _vec(seed, dim=128):
    v = np.random.RandomState(seed).randn(dim).astype(np.float32)
    return v / np.linalg.norm(v)


def _noisy(v, seed):
    return R.unit(v + 0.05 * np.random.RandomState(seed).randn(len(v)).astype(np.float32))


def _setup(tmp_path, now=None):
    st = MemoryStore(str(tmp_path / "m.db"), node="jetson")
    ana = st.put("person", {"display_name": "Ana", "relation": "family"})
    st.put("face_exemplar", {"person_id": ana, "emb": to_blob(_vec(1)), "source": "enroll"})
    rec = tmp_path / "rec"
    rec.mkdir()
    clock = [now or time.time()]
    rp = R.Replay(st, Gallery(st), None, str(rec), str(tmp_path / "thumbs"), clock=lambda: clock[0])
    return st, ana, rp, rec, clock


def _face(parent, v):
    return {"gie": 2, "parent": parent, "emb": v.astype(np.float16).tobytes()}


def _feed(rp, path, tracks, jpeg=b"\xff\xd8k", objs=()):
    """tracks: {parent: embedding}; 4 sampled frames each."""
    rp.cur = path
    for i in range(4):
        o = [_face(p, _noisy(v, 10 * p + i)) for p, v in tracks.items()] + list(objs)
        rp.on_msg(S.T_REPLAY, {"path": path, "cam": 1, "pts": 5.0 + i, "fn": i, "objs": o,
                               "jpeg": jpeg if i == 0 else None})


def test_seg_start_and_part_of_day():
    t = R.seg_start("/ssd/beni/rec/cam1_20260926-143000.ts")
    assert time.localtime(t)[:6] == (2026, 9, 26, 14, 30, 0)
    assert R.seg_start("/ssd/beni/rec/notes.txt") is None
    assert R.part_of_day(t) == "afternoon"


def test_pending_skips_fresh_old_done_and_foreign(tmp_path):
    st, _, rp, rec, clock = _setup(tmp_path)
    now = clock[0]
    for name, age in (("cam0_20260926-100000.ts", 3600), ("cam1_20260926-090000.ts", 7200),
                      ("cam0_20260926-110000.ts", 30), ("cam0_20260920-110000.ts", 7 * 86400),
                      ("cam0_20260926-080000.ts", 9000), ("other.ts", 3600)):
        p = rec / name
        p.write_bytes(b"\0")
        os.utime(str(p), (now - age, now - age))
    st.kv_set("replay_done", ["cam0_20260926-080000.ts"])
    assert [os.path.basename(p) for p in rp.pending()] == ["cam1_20260926-090000.ts", "cam0_20260926-100000.ts"]
    rp._mark_done(rp.pending()[0])
    assert len(rp.pending()) == 1


def test_known_person_gets_one_sighting(tmp_path):
    st, ana, rp, rec, _ = _setup(tmp_path)
    path = str(rec / "cam1_20260926-143000.ts")
    _feed(rp, path, {3: _vec(1)})
    rp.finish(path, {"frames": 900})
    eps = st.q("SELECT * FROM episode WHERE kind='sighting'")
    assert len(eps) == 1 and ana in eps[0]["people"] and "Ana was here around 14:30" in eps[0]["text"]
    _feed(rp, path, {3: _vec(1)})                          # same time again: no duplicate
    rp.finish(path)
    assert len(st.q("SELECT * FROM episode WHERE kind='sighting'")) == 1
    assert not rp.tracks and st.kv_get("replay_strangers") is None


def test_short_tracks_are_ignored(tmp_path):
    st, _, rp, rec, _ = _setup(tmp_path)
    path = str(rec / "cam1_20260926-143000.ts")
    rp.cur = path
    rp.on_msg(S.T_REPLAY, {"path": path, "pts": 1.0, "objs": [_face(3, _vec(2))]})
    rp.on_msg(S.T_REPLAY, {"path": "/elsewhere.ts", "pts": 1.0, "objs": [_face(3, _vec(2))]})
    assert len(rp.tracks[3]["embs"]) == 1
    rp.finish(path)
    assert not st.q("SELECT * FROM episode")


def test_stranger_on_two_days_becomes_visitor(tmp_path):
    st, _, rp, rec, _ = _setup(tmp_path)
    sent = []

    async def send(type_, **kw):
        sent.append((type_, kw))
    rp.link = SimpleNamespace(online=SimpleNamespace(is_set=lambda: True), send=send)
    st.kv_set("wanted_objects", ["keys", "umbrella"])
    stranger, other = _vec(7), _vec(8)

    async def go():
        p1 = str(rec / "cam1_20260925-150000.ts")
        _feed(rp, p1, {4: stranger, 5: other}, objs=[{"gie": 1, "cls": S.COCO.index("umbrella"), "conf": 0.6}])
        rp.finish(p1)
        cl = st.kv_get("replay_strangers")
        assert len(cl) == 2 and all(c["days"] == ["20260925"] and not c["ep"] for c in cl)
        assert st.q("SELECT label FROM object_sighting")[0]["label"] == "umbrella"
        assert st.kv_get("wanted_objects") == ["keys"]              # found locally: no longer asked of the brain
        p2 = str(rec / "cam0_20260926-163000.ts")
        _feed(rp, p2, {9: stranger})
        rp.finish(p2)
        await asyncio.sleep(0)
    asyncio.run(go())

    cl = {c["id"]: c for c in st.kv_get("replay_strangers")}
    rec_ = [c for c in cl.values() if len(c["days"]) == 2]
    assert len(cl) == 2 and len(rec_) == 1 and rec_[0]["ep"]
    ep = st.get("episode", rec_[0]["ep"])
    assert ep["kind"] == "unknown_visitor" and "afternoon" in ep["text"] and os.path.exists(ep["keyframe_path"])
    ask = st.kv_get("ask_unknown")
    assert ask["episode"] == ep["id"] and ask["cluster"] == rec_[0]["id"]
    assert sent[0][0] == "replay.flag" and sent[0][1]["id"] == ep["id"] and sent[0][1]["want"] == ["keys"]

    rp.on_result({"id": ep["id"], "caption": "a man holding keys at the door", "found": ["keys"]})
    ep = st.get("episode", ep["id"])
    assert ep["text"].endswith("Scene: a man holding keys at the door")
    assert {r["label"] for r in st.q("SELECT label FROM object_sighting")} == {"umbrella", "keys"}
    assert st.kv_get("wanted_objects") == []
    assert [f["label"] for f in st.kv_get("found_objects")] == ["umbrella", "keys"]
    rp.on_result({"id": "nope", "caption": "x"})              # unknown episode: ignored


def test_segment_round_trip_and_interrupt(tmp_path):
    st, _, rp, rec, _ = _setup(tmp_path)
    path = str(rec / "cam1_20260926-143000.ts")
    reqs = []

    async def go():
        async def request(ep, msg, timeout=2.0):
            reqs.append(msg)
            if msg.get("stop"):
                return {"ok": True}
            if msg["path"].endswith("bad.ts"):
                return {"ok": False, "err": "not a recording"}
            if msg["path"].endswith("busy.ts"):
                return {"ok": False, "err": "timeout"}

            async def later():
                await asyncio.sleep(0.01)
                _feed(rp, path, {3: _vec(1)})
                rp.on_msg(S.T_REPLAY, {"path": path, "done": True, "frames": 900, "ok": True})
            asyncio.ensure_future(later())
            return {"ok": True, "pending": 1}
        rp.request, rp.mode = request, "replay"
        assert await rp.segment(path) is True
        assert await rp.segment(str(rec / "cam0_20260926-000000-bad.ts")) is True
        assert await rp.segment(str(rec / "cam0_busy.ts")) is False
        rp.mode = "idle"
        rp._done = None

        async def never(ep, msg, timeout=2.0):
            reqs.append(msg)
            return {"ok": True}
        rp.request = never
        assert await rp.segment(path) is False
    asyncio.run(go())
    assert reqs[0] == {"op": "replay", "path": path, "stride": R.STRIDE} and reqs[-1] == {"op": "replay", "stop": 1}
    done = st.kv_get("replay_done")
    assert done == ["cam1_20260926-143000.ts", "cam0_20260926-000000-bad.ts"]
    assert len(st.q("SELECT * FROM episode WHERE kind='sighting'")) == 1 and rp.cur is None


def test_when_phrase():
    now = time.mktime((2026, 9, 27, 10, 0, 0, 0, 0, -1))
    at = lambda d, h: time.mktime((2026, 9, d, h, 0, 0, 0, 0, -1))   # noqa: E731
    assert when_phrase(at(27, 8), now) == "this morning"
    assert when_phrase(at(26, 15), now) == "yesterday afternoon"
    assert when_phrase(at(26, 23), now) == "last night"
    assert when_phrase(at(24, 19), now).endswith("evening")


def test_proactive_asks_about_visitor(tmp_path):
    st = MemoryStore(str(tmp_path / "m.db"), node="jetson")
    now = time.mktime((2026, 9, 27, 12, 0, 0, 0, 0, -1))       # midday: "yesterday afternoon", whatever the real time
    said = []

    class Voice:
        busy = False

        async def proactive(self, arm, ctx, text):
            said.append((arm, text))
    world = SimpleNamespace(people_present=lambda: ["p1"], battery=90, robot={}, name=lambda p: "Ana",
                            context=lambda: {"unknown_faces": 0})
    pro = Proactive(SimpleNamespace(quiet_hours=[]), world, Voice(), None, st, clock=lambda: now)
    pro.last_greet["p1"] = now
    assert not pro.candidates(now)
    st.kv_set("ask_unknown", {"episode": "e1", "t": now - 86400, "cluster": "c1"})
    (arm, person, extra), = pro.candidates(now)
    assert arm == "ask_visitor" and person == "p1" and extra["episode"] == "e1"
    asyncio.run(pro.execute(arm, person, extra))
    assert said[0][1].startswith("Ana, who was the person who visited yesterday") and st.kv_get("ask_unknown") is None
    st.kv_set("ask_unknown", {"episode": "e2", "t": now - 5 * 86400})
    assert not pro.candidates(now)                            # too old to ask about


def test_proactive_routine_memory_lost_item_wander(tmp_path):
    st = MemoryStore(str(tmp_path / "m.db"), node="jetson")
    now = time.mktime((2026, 9, 30, 18, 0, 0, 0, 0, -1))
    said, acts = [], []

    class Voice:
        busy, last_interaction = False, 0.0

        async def proactive(self, arm, ctx, text):
            said.append((arm, text))

    async def act(name, args):
        acts.append((name, args))
    people = ["p1"]
    world = SimpleNamespace(people_present=lambda: people, battery=90, robot={}, name=lambda p: "Ana", place=None,
                            context=lambda: {"unknown_faces": 0}, last_person_t=now - 3600, privacy=False)
    pro = Proactive(SimpleNamespace(quiet_hours=[]), world, Voice(), None, st, act=act, clock=lambda: now)
    pro.last_greet["p1"] = now
    for w in range(1, 4):                                   # Ana chats with Beni on Wednesday evenings
        st.add_episode("chat", people=["p1"], t_start=now - w * 7 * 86400, t_end=now - w * 7 * 86400)
    st.add_episode("Ana told Beni about her new job.", people=["p1"], importance=7,
                   t_start=now - 365 * 86400 + 3600, t_end=now - 365 * 86400 + 3600)
    routines.update(st, now - 60, since=now - 30 * 86400)
    st.kv_set("found_objects", [{"label": "keys", "t": now - 86400 - 3 * 3600}])
    cands = {c[0]: c for c in pro.candidates(now)}
    assert set(cands) == {"routine_nudge", "share_memory", "lost_item"}
    assert cands["share_memory"][2]["what"] == "Ana told Beni about her new job."
    for arm in ("lost_item", "share_memory"):
        asyncio.run(pro.execute(*cands[arm]))
    assert said == [("lost_item", "Ana, I think I spotted the keys yesterday afternoon."),
                    ("share_memory", "Ana, a year ago today: Ana told Beni about her new job.")]
    assert st.kv_get("found_objects") == [] and not pro.candidates(now)       # once a day each

    people.clear()
    assert not pro.candidates(now)                          # no places known yet
    st.put("place", {"name": "kitchen"})
    st.put("place", {"name": "dock", "kind": "dock"})
    (arm, person, extra), = pro.candidates(now)
    assert arm == "idle_wander" and extra == {"place": "kitchen"}

    async def run():
        await pro.execute(arm, person, extra)
        await asyncio.sleep(0)
    asyncio.run(run())
    assert acts == [("move_to", {"place": "kitchen"})] and len(said) == 2 and not pro.candidates(now)
    world.estop, pro.last_wander = True, 0.0
    assert not pro.candidates(now)                          # e-stop held: no wandering
    world.estop, world.battery = False, 40
    assert not pro.candidates(now)
