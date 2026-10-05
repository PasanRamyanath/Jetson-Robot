"""Memory store LWW, tombstones, forget-me and the DeltaSync handshake (§11.5) between two real stores."""
import asyncio

import numpy as np

from beni_common.memory import MemoryStore
from beni_common.memory.embed import HashEmbedder
from beni_common.memory.sync import DeltaSync, max_hlc


def wire(a, b):
    """Two DeltaSyncs whose send() delivers straight into the other side."""
    sa = sb = None

    async def send_a(type_, **kw):
        kw["type"] = type_
        await sb.on_frame(kw)
        return True

    async def send_b(type_, **kw):
        kw["type"] = type_
        await sa.on_frame(kw)
        return True
    sa = DeltaSync(a, send_a, "brain", b.node)
    sb = DeltaSync(b, send_b, "jetson", a.node)
    sa.reset("")
    sb.reset("")
    return sa, sb


async def drain(*syncs):
    for _ in range(20):
        if not sum([await s.push_once() for s in syncs]):
            return


def test_lww_and_tombstone(store_pair):
    a, b = store_pair
    pid = a.put("person", {"display_name": "Nimal"})
    delta, _ = a.changes_since("")
    assert b.apply_delta(delta)[0] >= 1
    b.put("person", {"id": pid, "display_name": "Nimal P."})      # newer write on the brain wins
    a.apply_delta(b.changes_since("")[0])
    assert a.get("person", pid)["display_name"] == "Nimal P."
    stale = dict(a.get("person", pid), display_name="old", hlc="0000000000001-0000-jetson")
    a.apply_delta({"person": [stale]})                               # older hlc loses
    assert a.get("person", pid)["display_name"] == "Nimal P."
    a.tombstone("person", pid)
    b.apply_delta(a.changes_since("")[0])
    assert b.get("person", pid) is None and b.get("person", pid, include_deleted=True)["deleted"] == 1


async def test_delta_sync_roundtrip(store_pair):
    a, b = store_pair
    for i in range(700):                                             # > batch: several acked rounds
        a.add_episode("episode %d" % i, importance=2)
    b.add_fact("p1", "likes", "tea", "p1 likes tea")
    sa, sb = wire(a, b)
    await drain(sa, sb)
    assert b.q("SELECT count(*) AS n FROM episode")[0]["n"] == 700
    assert a.facts_about("p1")[0]["object"] == "tea"
    assert a.peer_cursor("brain") >= max_hlc(a, "jetson")                # may skip past the peer's own rows
    assert await sa.push_once() == 0 and await sb.push_once() == 0   # converged, no echo


async def test_lost_ack_resends_idempotently(store_pair):
    a, b = store_pair
    a.add_episode("hello")
    sent = []

    async def lossy(type_, **kw):
        sent.append(type_)
        return True                                                  # delivered nowhere
    s = DeltaSync(a, lossy, "brain", "kaggle")
    s.reset("")
    assert await s.push_once() == 1
    assert await s.push_once() == 0                                  # one batch in flight
    s._sent_at -= 31                                                 # ack overdue ...
    s.nudge()
    task = asyncio.ensure_future(s.run())                            # ... so run() resends, even when nudged
    await asyncio.sleep(0.01)
    task.cancel()
    assert sent == ["memory.delta", "memory.delta"]
    s._inflight = None
    assert await s.push_once() == 1
    assert a.peer_cursor("brain") == ""


def test_hlc_never_goes_back_across_restarts(tmp_path):
    """A reboot with the clock behind the DB (no RTC, before NTP) still writes newer hlcs than every stored row."""
    from beni_common.hlc import HLC
    p = str(tmp_path / "m.db")
    a = MemoryStore(p)
    pid = a.put("person", {"display_name": "Ana"})
    old = a.get("person", pid)["hlc"]
    a.close()
    b = MemoryStore(p, hlc=HLC("jetson", clock=lambda: 1.0e9))      # 2001
    b.put("person", {"id": pid, "display_name": "Anna"})
    assert b.get("person", pid)["hlc"] > old and b.changes_since(old)[1] > old
    b.close()


def test_late_peer_episodes_reach_the_vector_index(store_pair):
    """The brain writes an episode, then a Jetson delta arrives with older hlcs: dense recall must still see it."""
    from beni_common.memory import Retriever
    a, b = store_pair
    e = HashEmbedder()
    old = a.add_episode("saw Ana in the kitchen", emb=e.doc("saw Ana in the kitchen"))
    r = Retriever(b, e)
    b.add_episode("talked about cricket", emb=e.doc("talked about cricket"))
    r.refresh()
    b.apply_delta(a.changes_since("")[0])
    r.refresh()
    assert old in r.index.pos and len(r.index) == 2


def test_forget_person(store_pair):
    a, _ = store_pair
    pid = a.put("person", {"display_name": "Kamal", "consent_face": 1})
    a.put("face_exemplar", {"person_id": pid, "emb": np.ones(512, np.float32).tobytes(), "created": 1})
    a.add_fact(pid, "likes", "cricket", "Kamal likes cricket")
    a.add_episode("Kamal talked about cricket", people=[pid])
    assert a.forget_person(pid) >= 3
    assert a.facts_about(pid) == []
    p = a.get("person", pid)
    assert p["do_not_learn"] == 1 and p["display_name"] is None
    ex = a.q("SELECT source FROM face_exemplar WHERE person_id=? AND deleted=0", (pid,))
    assert [e["source"] for e in ex] == ["do_not_learn"]


def test_supersede_keeps_history(store_pair):
    a, _ = store_pair
    old = a.add_fact("p1", "lives_in", "Kandy", "p1 lives in Kandy")
    new = a.supersede_fact(old, object="Colombo", text="p1 lives in Colombo")
    assert [f["object"] for f in a.facts_about("p1")] == ["Colombo"]
    assert a.get("fact", old)["superseded_by"] == new


def test_hash_embedder_is_normalised():
    e = HashEmbedder()
    v = e("where are my keys")
    assert abs(float(np.linalg.norm(v)) - 1.0) < 1e-4
    assert float(v @ e.doc("where are my keys")) > float(v @ e.doc("the weather is sunny"))


async def test_snapshot_into_mirror(tmp_path):
    from beni_agent.memory.sync import snapshot_bytes
    from beni_brain.memory.mirror import Mirror
    j = MemoryStore(str(tmp_path / "j.db"), node="jetson")
    for i in range(50):
        j.add_episode("walk %d" % i)
    m = Mirror(str(tmp_path / "k.db"), HashEmbedder())
    assert m.memory_since() is None
    data = await asyncio.to_thread(snapshot_bytes, j)
    assert await asyncio.to_thread(m.load_snapshot, data, {"hlc": max_hlc(j), "codec": "zlib"}) == 50
    assert m.memory_since() == max_hlc(j, "jetson")
    assert m.store.peer_cursor("jetson") == max_hlc(j)
